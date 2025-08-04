import torch
import torch.nn as nn
import timm
import torchvision
from torchvision import models
import os

class CustomClassifier(nn.Module):
    def __init__(self, input_features):
        super(CustomClassifier, self).__init__()
        self.classifier = nn.Sequential(
            # First dense layer with 256 neurons
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),  # 20% dropout

            # Second dense layer with 128 neurons
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch normalization again
            nn.ReLU(),

            # Final binary classifier
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(x)

print("CustomClassifier class defined")

# Model Class
class FeatureExtractionModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(FeatureExtractionModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)

        if isinstance(features, tuple):  # Some backbones return a tuple..........
            features = features[0]

        # Flatten (if necessary)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        # Classification using our head
        output = self.classifier(features)
        return output

print("FeatureExtractionModel class defined")

# Stacked Ensemble Model
class StackedEnsembleNet(nn.Module):
    def __init__(self, device):
        super(StackedEnsembleNet, self).__init__()
        self.device = device
        
        # Base models
        self.base_models = {}
        self._load_base_models()
        
        # Meta-learner - Hopefully what was intended in the paper
        # Input: concatenated predictions from base models (3 models × 2 outputs = 6 features)
        self.meta_learner = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, 2)
        ).to(device)
        
        print(f"✅ StackedEnsembleNet initialized with {len(self.base_models)} base models")
    
    def _load_base_models(self):
        """Load the three base models: InceptionV3, InceptionResNetV2, Xception"""
        
        # 1. InceptionV3
        try:
            inception_backbone = models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT, aux_logits=True)
            inception_backbone.fc = nn.Identity()
            inception_backbone.AuxLogits.fc = nn.Identity()
            for param in inception_backbone.parameters():
                param.requires_grad = False
            
            inception_classifier = CustomClassifier(input_features=2048)
            inception_model = FeatureExtractionModel(inception_backbone, inception_classifier).to(self.device)
            
            # Weights loading
            if os.path.exists('models/inception_v3_kidney_stone_model.pth'):
                inception_model.load_state_dict(torch.load('models/inception_v3_kidney_stone_model.pth', map_location=self.device))
                print("   📋 InceptionV3 weights loaded")
            else:
                print("   ⚠️ InceptionV3 weights not found, using pretrained backbone only")
            
            self.base_models['inception_v3'] = inception_model
        except Exception as e:
            print(f"   ❌ Failed to load InceptionV3: {e}")
        
        # 2. InceptionResNetV2
        try:
            resnet_backbone = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=0)
            for param in resnet_backbone.parameters():
                param.requires_grad = False
                
            resnet_classifier = CustomClassifier(input_features=resnet_backbone.num_features)
            resnet_model = FeatureExtractionModel(resnet_backbone, resnet_classifier).to(self.device)
            
            # Weights loading
            if os.path.exists('models/inceptionresnetv2_kidney_stone_model.pth'):
                resnet_model.load_state_dict(torch.load('models/inceptionresnetv2_kidney_stone_model.pth', map_location=self.device))
                print("   📋 InceptionResNetV2 weights loaded")
            else:
                print("   ⚠️ InceptionResNetV2 weights not found, using pretrained backbone only")
            
            self.base_models['inception_resnet_v2'] = resnet_model
        except Exception as e:
            print(f"   ❌ Failed to load InceptionResNetV2: {e}")
        
        # 3. Xception
        try:    
            xception_backbone = timm.create_model('xception', num_classes=0)
            for param in xception_backbone.parameters():
                param.requires_grad = False
                
            xception_classifier = CustomClassifier(input_features=xception_backbone.num_features)
            xception_model = FeatureExtractionModel(xception_backbone, xception_classifier).to(self.device)
            
            # Weights loading
            if os.path.exists('models/xception_kidney_stone_model.pth'):
                xception_model.load_state_dict(torch.load('models/xception_kidney_stone_model.pth', map_location=self.device))
                print("   📋 Xception weights loaded")
            else:
                print("   ⚠️ Xception weights not found, using pretrained backbone only")
            
            self.base_models['xception'] = xception_model
        except Exception as e:
            print(f"   ❌ Failed to load Xception: {e}")
    
    def forward(self, x):
        """
        Forward pass: 
        1. Get predictions from all base models
        2. Concatenate prediction vectors  
        3. Pass through meta-learner
        """
        # Base model predictions come first
        base_predictions = []
        
        for model_name, model in self.base_models.items():
            model.eval()
            with torch.no_grad():
                pred = torch.softmax(model(x), dim=1)
                base_predictions.append(pred)
        
        # Concatenate them predictions (feature representations as mentioned in the paper)
        if len(base_predictions) > 0:
            concatenated_features = torch.cat(base_predictions, dim=1)
        else:
            raise ValueError("No base models available for ensemble")
        
        # Pass through meta-learner (this gets trained...i hope)
        ensemble_output = self.meta_learner(concatenated_features)
        return ensemble_output
    
    def parameters(self):
        """Only return meta-learner parameters for training"""
        return self.meta_learner.parameters()

print("StackedEnsembleNet class defined")