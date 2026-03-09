import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess.flow_dataset import FlowDataset
# from models.hybrid_model import HybridFlowTab # Removed
from models.flow_switch import FlowSwitch
from config import Config
import os
import logging
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from datetime import datetime

def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_filename)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train():
    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------
    cfg = Config()
    
    # Setup Logger
    logger = setup_logger(cfg.log_dir)
    logger.info("Configuration loaded.")
    logger.info(f"Using device: {cfg.device}")

    # -------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------
    logger.info("Loading Flow Dataset (ISCX-VPN-2016)...")
    
    # Pass data_dir to load real data
    # Preprocess splits: Train 64%, Val 16%, Test 20%
    train_dataset = FlowDataset(cfg=cfg, split='train', data_dir=cfg.data_dir)
    val_dataset = FlowDataset(cfg=cfg, split='val', data_dir=cfg.data_dir)
    test_dataset = FlowDataset(cfg=cfg, split='test', data_dir=cfg.data_dir)
    
    # Recalculate num_classes in cfg is handled inside Dataset __init__ but we need to ensure model uses it.
    # Dataset updates cfg.num_classes in place.
    logger.info(f"Detected {cfg.num_classes} classes.")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=None, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # -------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------
    logger.info("Initializing FlowSwitch Model...")
    model = FlowSwitch(cfg).to(cfg.device)
    
    # -------------------------------------------------------------
    # 3. Optimization Setup
    # -------------------------------------------------------------
    # We optimize the whole model including transformer, stats extractor, and switchtab
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss() # Renamed to criterion

    # -------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------
    logger.info("Starting Training...")
    
    best_val_acc = 0.0
    
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        
        # Lists for metrics
        all_preds = []
        all_labels = []
        
        # Create an iterator for the second batch stream (for contrast/switch mechanism)
        train_iter = iter(train_loader)
        
        # We iterate through the loader manually or use enumerate
        # The key logic from previous buffer logic was likely aiming for pairs.
        # Let's simplify and make robust pair sampling.
        
        for i, (seq1, stat1, lbl1) in enumerate(train_loader):
            # Try to get a second batch
            try:
                seq2, stat2, lbl2 = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                seq2, stat2, lbl2 = next(train_iter)
            
            # Move to device
            seq1, stat1, lbl1 = seq1.to(cfg.device), stat1.to(cfg.device), lbl1.to(cfg.device)
            seq2, stat2, lbl2 = seq2.to(cfg.device), stat2.to(cfg.device), lbl2.to(cfg.device)
            
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with pairs
            # outputs = {logits1, logits2, recon_loss}
            outputs = model(seq1, stat1, seq2, stat2)
            
            logits1 = outputs['logits1']
            logits2 = outputs['logits2']
            recon_loss = outputs['recon_loss']
            
            # Classification Loss
            loss_cls = criterion(logits1, lbl1) + criterion(logits2, lbl2)
            
            # Total loss
            loss = recon_loss + cfg.loss_alpha * loss_cls
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            # Collect predictions for metrics
            _, pred1 = torch.max(logits1, 1)
            _, pred2 = torch.max(logits2, 1)
            
            all_preds.extend(pred1.cpu().numpy())
            all_preds.extend(pred2.cpu().numpy())
            all_labels.extend(lbl1.cpu().numpy())
            all_labels.extend(lbl2.cpu().numpy())
            
        # Calculate Epoch Metrics
        epoch_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds) * 100
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for seq_input, stat_input, labels in val_loader:
                seq_input, stat_input, labels = seq_input.to(cfg.device), stat_input.to(cfg.device), labels.to(cfg.device)
                
                # Validation forward (single path)
                # Ensure model.forward handles single input correctly (it does based on previous checks)
                outputs = model(seq_input, stat_input) 
                
                # Check output format, might be dict or tensor depending on implementation details
                if isinstance(outputs, dict):
                     logits = outputs['logits']
                else:
                     logits = outputs
                
                _, predicted = torch.max(logits, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds) * 100
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model if needed
            # torch.save(model.state_dict(), os.path.join(cfg.log_dir, 'best_model.pth'))
        
        logger.info(f"Epoch [{epoch+1}/{cfg.epochs}] "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}% F1: {train_f1:.4f} | "
                    f"Val Acc: {val_acc:.2f}% F1: {val_f1:.4f}")
              
    logger.info("Training Finished.")
    
    # -------------------------------------------------------------
    # 5. Testing Loop
    # -------------------------------------------------------------
    logger.info("Starting Testing...")
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for seq_input, stat_input, labels in test_loader:
            seq_input, stat_input, labels = seq_input.to(cfg.device), stat_input.to(cfg.device), labels.to(cfg.device)
            outputs = model(seq_input, stat_input)
            
            if isinstance(outputs, dict):
                 logits = outputs['logits']
            else:
                 logits = outputs
            
            _, predicted = torch.max(logits, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(test_labels, test_preds) * 100
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    logger.info("=========================================")
    logger.info(f"Test Results: Accuracy: {test_acc:.2f}% | F1 Score: {test_f1:.4f}")
    logger.info("=========================================")
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
