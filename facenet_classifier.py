import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetClassifier(nn.Module):
    def __init__(self, num_emotions=8, freeze_facenet=True):
        super(FaceNetClassifier, self).__init__()
        # 1) Load FaceNet
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        
        if freeze_facenet:
            for param in self.facenet.parameters():
                param.requires_grad = False
        
        # 2) Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),    # first dense
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),    # second dense
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_emotions)  # final logits
        )

    def forward(self, x):
        # If facenet is in eval mode and freeze_facenet=True, it uses no_grad().
        # For partial fine-tuning, we set freeze_facenet=False & call train() on it 
        # or just manually set requires_grad to True for certain layers.
        with torch.no_grad() if not self.facenet.training else torch.enable_grad():
            embeddings = self.facenet(x)  # shape: (batch_size, 512)
        out = self.classifier(embeddings)
        return out