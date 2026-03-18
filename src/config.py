
import torch

# Main config class

class Config:
    def __init__(self):
        # Data settings
        self.min_session_length = 3
        self.min_item_frequency = 5
        self.test_days = 1
        self.data_fraction = 0.25
        
        # Model settings
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.n_gnn_layers = 3
        self.n_gru_layers = 1
        self.n_attention_heads = 4 
        self.dropout = 0.2
        
        # Training settings
        self.batch_size = 512
        self.epochs = 30
        self.learning_rate = 0.0005  
        self.weight_decay = 1e-5
        self.max_grad_norm = 5.0
        self.patience = 5
        self.lr_step_size = 3
        self.lr_gamma = 0.1
        
        # Evaluation settings
        self.top_k = 20
        
        # Device setting
        self.device = "auto"
        
        # Auto device detection
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self):
        return {
            "min_session_length": self.min_session_length,
            "min_item_frequency": self.min_item_frequency,
            "test_days": self.test_days,
            "data_fraction": self.data_fraction,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "n_gnn_layers": self.n_gnn_layers,
            "n_gru_layers": self.n_gru_layers,
            "n_attention_heads": self.n_attention_heads,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "patience": self.patience,
            "lr_step_size": self.lr_step_size,
            "lr_gamma": self.lr_gamma,
            "top_k": self.top_k,
            "device": self.device
        }

# Default config
def get_default_config():
    return Config()


def print_device_info():
    #Print GPU info
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("GPU not found, using CPU")
