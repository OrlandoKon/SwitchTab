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
    print("Loading Flow Dataset (Mock)...")
    # Using mock data generation by passing data=None
    train_dataset = FlowDataset(cfg=cfg, data=None, is_train=True)
    test_dataset = FlowDataset(cfg=cfg, data=None, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
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
        
        for i, (seq_input, stat_input, labels) in enumerate(train_loader):
            seq_input = seq_input.to(cfg.device)
            stat_input = stat_input.to(cfg.device)
            labels = labels.to(cfg.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Split batch into two halves for x1 and x2
            batch_size = seq_input.size(0)
            split_idx = batch_size // 2
            
            if split_idx == 0:
                continue
                
            seq1, seq2 = seq_input[:split_idx], seq_input[split_idx:2*split_idx]
            stat1, stat2 = stat_input[:split_idx], stat_input[split_idx:2*split_idx]
            lbl1, lbl2 = labels[:split_idx], labels[split_idx:2*split_idx]

            # Forward pass with pairs
            outputs = model(seq1, stat1, seq2, stat2)
            
            logits1 = outputs['logits1']
            logits2 = outputs['logits2']
            recon_loss = outputs['recon_loss']
            
            # Calculate classification loss
            cls_loss = 0.5 * (criterion_cls(logits1, lbl1) + criterion_cls(logits2, lbl2))
            
            # Total loss: L_cls + lambda * L_recon
            loss = recon_loss + cfg.loss_alpha * cls_loss
            
            # Backward pass
            loss.backward()
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
        
        print(f"Epoch [{epoch+1}/{cfg.epochs}] "
              f"Loss: {epoch_loss:.4f} (Cls: {running_cls/len(train_loader):.4f}, Rec: {running_rec/len(train_loader):.4f}) "
              f"Acc: {epoch_acc:.2f}%")
              
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
