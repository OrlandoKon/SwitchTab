import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess.flow_dataset import FlowDataset
# from models.hybrid_model import HybridFlowTab # Removed
from models.flow_switch import FlowSwitch
from config import Config

def train():
    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------
    cfg = Config()
    
    # Check device
    print(f"Using device: {cfg.device}")

    # -------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------
    print("Loading Flow Dataset (ISCX-VPN-2016)...")
    
    # Pass data_dir to load real data
    # Preprocess splits: Train 64%, Val 16%, Test 20%
    train_dataset = FlowDataset(cfg=cfg, split='train', data_dir=cfg.data_dir)
    val_dataset = FlowDataset(cfg=cfg, split='val', data_dir=cfg.data_dir)
    test_dataset = FlowDataset(cfg=cfg, split='test', data_dir=cfg.data_dir)
    
    # Recalculate num_classes in cfg is handled inside Dataset __init__ but we need to ensure model uses it.
    # Dataset updates cfg.num_classes in place.
    print(f"Detected {cfg.num_classes} classes.")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=None, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # -------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------
    print("Initializing FlowSwitch Model...")
    model = FlowSwitch(cfg).to(cfg.device)
    
    # -------------------------------------------------------------
    # 3. Optimization Setup
    # -------------------------------------------------------------
    # We optimize the whole model including transformer, stats extractor, and switchtab
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion_cls = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------
    print("Starting Training...")
    
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_rec = 0.0
        correct = 0
        total = 0
        
        # Create an iterator for the second batch stream
        loader_iter = iter(train_loader)
        
        for i, (seq1, stat1, lbl1) in enumerate(train_loader):
            # Try to get the next batch for x2, if exhausted, restart iterator
            try:
                seq2, stat2, lbl2 = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                seq2, stat2, lbl2 = next(loader_iter)
            
            # Ensure we are not pairing identical batches if possible (though with shuffle=True it's probabilistic)
            # In standard training, we just want two different samples.
            
            # Move to device
            seq1, stat1, lbl1 = seq1.to(cfg.device), stat1.to(cfg.device), lbl1.to(cfg.device)
            seq2, stat2, lbl2 = seq2.to(cfg.device), stat2.to(cfg.device), lbl2.to(cfg.device)
            
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with pairs from different batches
            outputs = model(seq1, stat1, seq2, stat2)
            
            logits1 = outputs['logits1']
            logits2 = outputs['logits2']
            recon_loss = outputs['recon_loss']
            
            # Calculate classification loss according to: L_cls = - (y1 log(y^1) + y2 log(y^2))
            cls_loss = criterion_cls(logits1, lbl1) + criterion_cls(logits2, lbl2)
            
            # Total loss: L_recon + alpha * L_cls
            loss = recon_loss + cfg.loss_alpha * cls_loss
            
            # Loss backprop
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            running_cls += cls_loss.item()
            running_rec += recon_loss.item()
            
            _, predicted1 = torch.max(logits1.data, 1)
            _, predicted2 = torch.max(logits2.data, 1)
            
            total += lbl1.size(0) + lbl2.size(0)
            correct += (predicted1 == lbl1).sum().item() + (predicted2 == lbl2).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / (total + 1e-9)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for seq_input, stat_input, labels in val_loader:
                seq_input, stat_input, labels = seq_input.to(cfg.device), stat_input.to(cfg.device), labels.to(cfg.device)
                outputs = model(seq_input, stat_input)
                # Ensure the model returns 'logits' when not training in pair mode
                logits = outputs.get('logits', None)
                if logits is None: # Should not happen with current FlowSwitch.forward logic
                    logits = outputs['logits1'] 
                
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100.0 * val_correct / (val_total + 1e-9)
        
        print(f"Epoch [{epoch+1}/{cfg.epochs}] "
              f"Loss: {epoch_loss:.4f} (Cls: {running_cls/len(train_loader):.4f}, Rec: {running_rec/len(train_loader):.4f}) "
              f"Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
              
    print("Training Finished.")
    
    # -------------------------------------------------------------
    # 5. Testing Loop
    # -------------------------------------------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seq_input, stat_input, labels in test_loader:
            seq_input = seq_input.to(cfg.device)
            stat_input = stat_input.to(cfg.device)
            labels = labels.to(cfg.device)
            outputs = model(seq_input, stat_input)
            _, predicted = torch.max(outputs['logits'].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
