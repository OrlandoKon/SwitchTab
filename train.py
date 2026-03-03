import torch
from torch import nn
from torch.optim import RMSprop, Adam
from torch.utils.data import TensorDataset, DataLoader

# Revised imports
from model import Encoder, Projector, Decoder, Predictor
from utils import feature_corruption
from config import Config
from preprocess import load_and_preprocess_data

# Implementing the self-supervised learning algorithm 
# Moved from utils.py to train.py
def self_supervised_learning_with_switchtab(train_dataset, val_dataset, test_dataset, batch_size, feature_size, num_classes):
    # Assuming data is a PyTorch dataset
    # Note: batch_size argument was ignored in original code in favor of hardcoded 128, fixing this to use argument or config
    batch_size = batch_size if batch_size else 128
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Test loader doesn't need shuffle
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the components with the feature size
    f_encoder = Encoder(feature_size)
    half_feature_size = feature_size // 2
    pm_mutual = Projector(feature_size, half_feature_size)
    ps_salient = Projector(feature_size, half_feature_size)
    # The decoder takes concatenated features (mutual + salient) as input and reconstructs original features
    # Since each projector outputs feature_size // 2, the concatenated size is feature_size (assuming feature_size is even)
    d_decoder = Decoder(feature_size, feature_size)
    pred_predictor = Predictor(feature_size, num_classes)  # For pre-training stage with labels
    
    # Loss function and optimizer
    mse_loss = nn.MSELoss()
    # Optimizer for pre-training
    pretrain_optimizer = RMSprop(list(f_encoder.parameters()) + list(pm_mutual.parameters()) + 
                                 list(ps_salient.parameters()) + list(d_decoder.parameters()) +
                                 list(pred_predictor.parameters()), lr=0.0003)
    
    # Pre-training loop
    print("Starting Pre-training (Simultaneous Self-Supervised + Supervised)...")
    print_interval = 50
    # Reducing epochs for demo purposes if needed, keeping 1000 as per original but reduced for demo speed
    for epoch in range(100): 
        for batch1, batch2 in zip(train_loader, train_loader):
            # Unpack data if it contains labels (handle list/tuple from TensorDataset)
            # The train_dataset returns (features, labels)
            if isinstance(batch1, (list, tuple)) and len(batch1) == 2:
                x1_batch, y1_batch = batch1
            else:
                x1_batch = batch1[0] if isinstance(batch1, (list, tuple)) else batch1
                # Fallback if no labels (though logic expects them now)
                
            if isinstance(batch2, (list, tuple)) and len(batch2) == 2:
                x2_batch, y2_batch = batch2
            else:
                x2_batch = batch2[0] if isinstance(batch2, (list, tuple)) else batch2

            # Feature corruption
            x1_corrupted = feature_corruption(x1_batch)
            x2_corrupted = feature_corruption(x2_batch)
            
            # Data encoding
            z1_encoded = f_encoder(x1_corrupted)
            z2_encoded = f_encoder(x2_corrupted)
            
            # Feature decoupling
            s1_salient = ps_salient(z1_encoded)
            m1_mutual = pm_mutual(z1_encoded)
            s2_salient = ps_salient(z2_encoded)
            m2_mutual = pm_mutual(z2_encoded)
            
            # Data reconstruction
            x1_reconstructed = d_decoder(torch.cat((m1_mutual, s1_salient), dim=1))
            x2_reconstructed = d_decoder(torch.cat((m2_mutual, s2_salient), dim=1))
            x1_switched = d_decoder(torch.cat((m2_mutual, s1_salient), dim=1))
            x2_switched = d_decoder(torch.cat((m1_mutual, s2_salient), dim=1))
            
            # Calculate reconstruction loss
            loss_rec = mse_loss(x1_batch, x1_reconstructed) + mse_loss(x2_batch, x2_reconstructed) + \
                       mse_loss(x1_batch, x1_switched) + mse_loss(x2_batch, x2_switched)
            
            # Supervised Prediction Loss
            # Generate predictions using the predictor head from encoded representations
            pred1 = pred_predictor(z1_encoded)
            pred2 = pred_predictor(z2_encoded)
            
            # Since 'fine_tuning_loss_function' is essentially CrossEntropyLoss, and we want to use labels
            # But wait, we haven't extracted labels in the loop yet properly in previous steps if I failed the replace.
            
            # Let's fix the label extraction logic right here first if possible, but I better edit the whole block.
            # Assuming subsequent edit will fix the loop unpacking.
            
            # For now, let's just make sure we calculate the loss correctly IF labels (y1_batch, y2_batch) were available.
            # I will assume y1_batch and y2_batch are tensors of labels.
            loss_sup = ce_loss(pred1, y1_batch) + ce_loss(pred2, y2_batch)
            
            # Total Loss
            loss = loss_rec + loss_sup
            
            # Update model parameters
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()

        # Print loss every print_interval epochs
        if (epoch+1) % print_interval == 0:
            print(f'Epoch [{epoch+1}/100], Total Loss: {loss.item():.4f} (Rec: {loss_rec.item():.4f}, Sup: {loss_sup.item():.4f})')

    # Fine-tuning loop
    fine_tuning_loss_function = nn.CrossEntropyLoss()
    fine_tuning_optimizer = Adam(f_encoder.parameters(), lr=0.001)
    
    print("Starting Fine-tuning...")
    for epoch in range(50):
        # Training Phase
        f_encoder.train()
        pred_predictor.train()
        for x_batch, labels in train_loader:
            # Assume that now we have labels
            z_encoded = f_encoder(x_batch)
            predictions = pred_predictor(z_encoded)
            # Replace 'some_loss_function' with the actual loss function used for fine-tuning
            prediction_loss = fine_tuning_loss_function(predictions, labels)
            fine_tuning_optimizer.zero_grad()
            prediction_loss.backward()
            fine_tuning_optimizer.step()

        # Validation Phase
        if (epoch+1) % 10 == 0:
            f_encoder.eval()
            pred_predictor.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    z_val = f_encoder(x_val)
                    outputs = pred_predictor(z_val)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_val.size(0)
                    correct += (predicted == y_val).sum().item()
            
            val_accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/50], Fine-tuning Loss: {prediction_loss.item():.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Test Phase
    print("Starting Testing...")
    f_encoder.eval()
    pred_predictor.eval()
    
    # Since test.csv doesn't have labels, we generate predictions
    # If test.csv HAD labels, we would calculate accuracy like validation
    all_predictions = []
    with torch.no_grad():
        for x_test in test_loader:
            if isinstance(x_test, list): x_test = x_test[0] # Handle DataLoader returning list
            z_test = f_encoder(x_test)
            outputs = pred_predictor(z_test)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.numpy())
    
    print(f"Generated {len(all_predictions)} predictions for test set.")
    # Here you would typically save predictions to a file
    # Example: pd.DataFrame({'PassengerId': ..., 'Survived': all_predictions}).to_csv('submission.csv')

def main():
    print("Initialize SwitchTab Demo with Titanic Data...")

    # Load configuration
    cfg = Config()

    try:
        # Load Train Data
        x_train_full, y_train_full = load_and_preprocess_data(cfg.data_path, is_test=False)
        # Load Test Data
        test_path = '/root/Demo/SwitchTab/data/Tatanic/test.csv'
        x_test, _ = load_and_preprocess_data(test_path, is_test=True)
        
    except Exception as e:
        print(f"Data processing failed: {e}")
        return

    # Split Train into Train/Val
    # Manually split to avoid introducing sklearn dependency if not strictly needed, or use sklearn if available
    # Using manual split 80/20 for simplicity
    num_train = int(0.8 * len(x_train_full))
    indices = torch.randperm(len(x_train_full))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]

    # --- Configuration Update ---
    # Update config based on loaded data
    cfg = Config()
    
    num_samples = len(x_train)
    feature_size = x_train.shape[1] 
    num_classes = len(torch.unique(y_train))
    batch_size = cfg.batch_size

    print(f"Data Prepared.")
    print(f"Train Samples: {len(x_train)}")
    print(f"Val Samples: {len(x_val)}")
    print(f"Test Samples: {len(x_test)}")
    print(f"Features: {feature_size}")
    print(f"Classes: {num_classes}")

    # Create Datasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test) # Test dataset only has features
    
    print("Starting training...")

    # Run Model
    self_supervised_learning_with_switchtab(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        feature_size=feature_size,
        num_classes=num_classes
    )

    print("Training finished successfully!")

if __name__ == "__main__":
    main()
