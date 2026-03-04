import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess.flow_dataset import FlowDataset
from models.hybrid_model import HybridFlowTab

def train():
    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    LAMBDA_RECON = 0.5
    # The output dimension of the classifier must match this NUM_CLASSES
    # But wait, SwitchTabModel expects num_classes.
    NUM_CLASSES = 2
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------
    print("Loading Flow Dataset (Mock)...")
    # Using mock data generation by passing data=None
    train_dataset = FlowDataset(data=None, is_train=True)
    test_dataset = FlowDataset(data=None, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # -------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------
    print("Initializing HybridFlowTab Model...")
    model = HybridFlowTab(num_classes=NUM_CLASSES).to(device)
    
    # -------------------------------------------------------------
    # 3. Optimization Setup
    # -------------------------------------------------------------
    # We optimize the whole model including transformer, stats extractor, and switchtab
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_cls = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_rec = 0.0
        correct = 0
        total = 0
        
        for i, (seq_input, stat_input, labels) in enumerate(train_loader):
            seq_input = seq_input.to(device)
            stat_input = stat_input.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            # Returns dict: {'logits': ..., 'recon_loss': ...}
            outputs = model(seq_input, stat_input)
            
            logits = outputs['logits']
            recon_loss = outputs['recon_loss']
            
            # Calculate classification loss
            cls_loss = criterion_cls(logits, labels)
            
            # Total loss: L_cls + lambda * L_recon
            loss = cls_loss + LAMBDA_RECON * recon_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            running_cls += cls_loss.item()
            running_rec += recon_loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / (total + 1e-9)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
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
            seq_input = seq_input.to(device)
            stat_input = stat_input.to(device)
            labels = labels.to(device)
            outputs = model(seq_input, stat_input)
            _, predicted = torch.max(outputs['logits'].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
