import torch
import torch.nn as nn

class DeepSkinDiseaseANN(nn.Module):
    """
    Deep Artificial Neural Network (ANN/MLP) for Skin Disease Classification.
    Trained on 78 clinical and demographic features from the PAD-UFES-20 dataset.
    Classifies lesions into 6 diagnostic classes:
    0: ACK (Actinic Keratosis)
    1: BCC (Basal Cell Carcinoma)
    2: MEL (Melanoma)
    3: NEV (Nevus)
    4: SCC (Squamous Cell Carcinoma)
    5: SEK (Seborrheic Keratosis)
    """
    def __init__(self, input_size=78, num_classes=6):
        super(DeepSkinDiseaseANN, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output Layer
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def load_trained_model(model_path, device='cpu'):
    """
    Utility function to load the trained PyTorch weights into DeepSkinDiseaseANN.
    """
    model = DeepSkinDiseaseANN(input_size=78, num_classes=6)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
