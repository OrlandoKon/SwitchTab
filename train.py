import torch
from torch.utils.data import TensorDataset
from utils import self_supervised_learning_with_switchtab
from config import Config
from preprocess import load_and_preprocess_data

def main():
    print("Initialize SwitchTab Demo with Titanic Data...")

    # Load configuration
    cfg = Config()

    try:
        x_train, y_train = load_and_preprocess_data(cfg.data_path)
    except Exception as e:
        print(f"Data processing failed: {e}")
        return

    # --- Configuration Update ---
    # Update config based on loaded data
    cfg = Config()
    
    num_samples = len(x_train)
    feature_size = x_train.shape[1] 
    num_classes = len(torch.unique(y_train))
    batch_size = cfg.batch_size

    print(f"Data Prepared.")
    print(f"Samples: {num_samples}")
    print(f"Features: {feature_size}")
    print(f"Classes: {num_classes}")

    # Create Dataset
    dataset = TensorDataset(x_train, y_train)
    
    print("Starting training...")

    # Run Model
    self_supervised_learning_with_switchtab(
        data=dataset,
        batch_size=batch_size,
        feature_size=feature_size,
        num_classes=num_classes
    )

    print("Training finished successfully!")

if __name__ == "__main__":
    main()
