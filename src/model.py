import torch
import torch.nn as nn
from torchvision import models

class AttentionPooling(nn.Module):
    """
    An Attention Pooling layer that learns to weight different parts of the feature map.
    """
    def __init__(self, in_features):
        super(AttentionPooling, self).__init__()
        # A small network to compute attention scores
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_features // 8, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape: [batch_size, num_features, height, width]

        # Compute attention scores (unnormalized)
        # Shape: [batch_size, 1, height, width]
        attention_scores = self.attention_net(x)

        # Reshape scores and apply softmax to get attention weights
        # Shape: [batch_size, height * width]
        attention_weights = torch.softmax(attention_scores.view(x.size(0), -1), dim=1)

        # Reshape feature map to apply weights
        # Shape: [batch_size, num_features, height * width]
        features = x.view(x.size(0), x.size(1), -1)
        # Shape: [batch_size, height * width, num_features] for batch matrix multiplication
        features = features.permute(0, 2, 1)

        # Compute weighted average of features
        # Unsqueeze adds a dimension for broadcasting: [batch_size, 1, height * width]
        # bmm: [batch_size, 1, height * width] @ [batch_size, height * width, num_features]
        # Result shape: [batch_size, 1, num_features]
        weighted_features = torch.bmm(attention_weights.unsqueeze(1), features)

        # Remove the extra dimension
        # Shape: [batch_size, num_features]
        return weighted_features.squeeze(1)
    

class SpectrogramResNet(nn.Module):
    """
    A ResNet based model for spectrogram classification using Attention Pooling
    """
    def __init__(self, num_classes=1, dropout_rate=0.4):
        super(SpectrogramResNet, self).__init__()

        # Load pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze all parameters in pre-trained model
        for param in resnet.parameters():
            param.requires_grad = False

        # Extract the feature extractor part of ResNet (all layers except avgpool and fc)
        # ResNet-50's feature extractor outputs 2048 features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.num_features = resnet.fc.in_features

        # Replace pooling layer with AttentionPooling layer
        self.pool = AttentionPooling(in_features=self.num_features)

        # Define new classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, self.num_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features // 2, num_classes)
        )

        def forward(self, x):
            """
            Defines the forward pass of the mode.
            """
            # Extract features from backbone
            # Input x shape: [batch_size, 3, 224, 224]
            # Output features shape: [batch_size, 2048, 7, 7]
            features = self.backbone(x)

            # Apply attention pooling
            # Output pooled_features shape: [batch_size, 2048]
            pooled_features = self.pool(features)

            # Get final classification scores (logits)
            # Output logits shape: [batch_size, 1]
            logits = self.classifier(pooled_features)

            # For binary classification with BCEWithLogitsLoss, want a squeezed output
            # Output shape: [batch_size]
            return logits.squeeze(-1)


# Block to test cript
if __name__ == '__main__':
    # Create a dummy input tensor to test the model
    dummy_input = torch.randn(4, 3, 224, 224) # (batch_size, channels, height, width)
    
    # Instantiate the model
    model = SpectrogramResNet()
    
    # Check that model parameters are frozen correctly
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"
    
    print("Backbone layers are correctly frozen.")
    
    # Pass the dummy input through the model
    output = model(dummy_input)
    
    # Print the output shape to verify it's correct
    print(f"\nModel instantiated successfully.")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be torch.Size([4])
    assert output.shape == torch.Size([4])
    print("Output shape is correct for binary classification.")