from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import os
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
import io
from typing import Dict, Optional, List

# Add the current directory to Python path to import architectures
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from architectures import CustomClassifier, FeatureExtractionModel, StackedEnsembleNet

app = FastAPI(
    title="SEN-D Kidney Stone Detection API",
    description="AI-powered kidney stone detection from CT scan images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "https://sen-d.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
model = None
device = None

class PredictionResponse(BaseModel):
    ensemble: Dict
    individual_models: Dict
    processing_time: float
    num_models: int
    success: bool
    message: str

class GradCAM:
    """Grad-CAM implementation for generating heatmaps"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        # Find the target layer
        target_layer = self._find_target_layer()
        if target_layer is not None:
            self.hooks.append(target_layer.register_forward_hook(forward_hook))
            self.hooks.append(target_layer.register_backward_hook(backward_hook))
    
    def _find_target_layer(self):
        """Find the target layer by name"""
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        
        # If exact name not found, try to find the last convolutional layer
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d)):
                conv_layers.append((name, module))
        
        if conv_layers:
            return conv_layers[-1][1]
        
        return None
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Class Activation Map"""
        try:
            # Forward pass
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            target = output[0, class_idx]
            target.backward(retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                return None
            
            # Generate CAM
            gradients = self.gradients[0]  # Remove batch dimension
            activations = self.activations[0]  # Remove batch dimension
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=[1, 2])
            
            # Weighted combination of activation maps
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # Apply ReLU
            cam = torch.relu(cam)
            
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam.detach().cpu().numpy()
            
        except Exception as e:
            print(f"Error in generate_cam: {e}")
            return None
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()

def create_heatmap_overlay(original_image, cam, alpha=0.4):
    """Create heatmap overlay on original image"""
    try:
        # Ensure CAM is 2D
        if len(cam.shape) > 2:
            cam = np.squeeze(cam)
        
        # Resize CAM to match input image size
        cam_resized = cv2.resize(cam, (299, 299))
        
        # Convert original image to numpy array
        if isinstance(original_image, Image.Image):
            # Ensure the image is in RGB mode before conversion
            rgb_image = original_image.convert('RGB')
            img_array = np.array(rgb_image.resize((299, 299)))
        else:
            img_array = original_image
        
        # Ensure img_array always has 3 channels
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Single channel
            img_array = np.repeat(img_array, 3, axis=2)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Remove alpha channel
        
        # Create heatmap
        heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Ensure both arrays have the same shape
        if img_array.shape != heatmap.shape:
            # Resize heatmap to match img_array if needed
            if img_array.shape[:2] != heatmap.shape[:2]:
                heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
        
        return overlay, heatmap
        
    except Exception as e:
        print(f"Error in create_heatmap_overlay: {e}")
        return None, None

def get_target_layer_name(model_name):
    """Get appropriate target layer name for different architectures"""
    layer_mapping = {
        'inception_v3': 'backbone.Mixed_7c',
        'inception_resnet_v2': 'backbone.conv2d_7b',
        'xception': 'backbone.conv4'
    }
    return layer_mapping.get(model_name, 'backbone.features')

def load_model():
    """Load and cache the model"""
    global model, device
    
    if model is not None:
        return model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = StackedEnsembleNet(device)
        if os.path.exists('models/stacked_ensemble_meta_learner.pth'):
            model.meta_learner.load_state_dict(
                torch.load('models/stacked_ensemble_meta_learner.pth', map_location=device)
            )
            print("✅ StackedEnsembleNet loaded successfully!")
        else:
            print("⚠️ No saved weights found, using pre-trained backbones only")
        
        model.eval()
        return model, device
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    try:
        # Convert to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_array)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def predict_all_models(ensemble_model, device, image):
    """Make predictions on all models (base models + ensemble) with Grad-CAM"""
    try:
        input_tensor = preprocess_image(image).to(device)
        class_names = ["Kidney_stone", "Normal"]
        results = {}
        
        start_time = time.time()
        
        # 1. Get predictions from individual base models with Grad-CAM
        individual_results = {}
        base_predictions = []
        
        for model_name, base_model in ensemble_model.base_models.items():
            base_model.eval()
            with torch.no_grad():
                outputs = base_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Generate Grad-CAM for this model
            target_layer = get_target_layer_name(model_name)
            gradcam = GradCAM(base_model, target_layer)
            
            # Enable gradients for Grad-CAM
            input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
            cam = gradcam.generate_cam(input_tensor_grad, predicted_class)
            gradcam.remove_hooks()
            
            # Create heatmap overlay
            overlay_base64 = None
            heatmap_base64 = None
            if cam is not None:
                try:
                    overlay, heatmap = create_heatmap_overlay(image, cam)
                    if overlay is not None:
                        overlay_base64 = image_to_base64(overlay)
                    if heatmap is not None:
                        heatmap_base64 = image_to_base64(heatmap)
                except Exception as e:
                    print(f"Failed to create heatmap for {model_name}: {e}")
            
            individual_results[model_name] = {
                "prediction": class_names[predicted_class],
                "confidence": float(probabilities[predicted_class]),
                "probabilities": {
                    "Kidney_stone": float(probabilities[0]),
                    "Normal": float(probabilities[1])
                },
                "gradcam_overlay": overlay_base64,
                "gradcam_heatmap": heatmap_base64
            }
            base_predictions.append(probabilities)
        
        # 2. Get ensemble prediction (no Grad-CAM for ensemble as it's a meta-learner)
        ensemble_model.eval()
        with torch.no_grad():
            ensemble_outputs = ensemble_model(input_tensor)
            ensemble_probabilities = torch.softmax(ensemble_outputs, dim=1)[0]
            ensemble_predicted_class = torch.argmax(ensemble_outputs, dim=1).item()
            
            ensemble_result = {
                "prediction": class_names[ensemble_predicted_class],
                "confidence": float(ensemble_probabilities[ensemble_predicted_class]),
                "probabilities": {
                    "Kidney_stone": float(ensemble_probabilities[0]),
                    "Normal": float(ensemble_probabilities[1])
                }
            }
        
        processing_time = time.time() - start_time
        
        return {
            "ensemble": ensemble_result,
            "individual_models": individual_results,
            "processing_time": processing_time,
            "num_models": len(ensemble_model.base_models) + 1,
            "success": True,
            "message": "Prediction completed successfully"
        }
        
    except Exception as e:
        return {
            "ensemble": {},
            "individual_models": {},
            "processing_time": 0,
            "num_models": 0,
            "success": False,
            "message": f"Prediction failed: {str(e)}"
        }

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, device
    model, device = load_model()
    if model is None:
        print("Failed to load model on startup!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SEN-D Kidney Stone Detection API",
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "num_base_models": len(model.base_models) if model else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict kidney stone presence from CT scan image
    
    - **file**: CT scan image file (JPG, PNG, JPEG)
    
    Returns predictions from ensemble model and individual base models with Grad-CAM visualizations
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make predictions
        results = predict_all_models(model, device, image)
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
async def get_models():
    """Get information about loaded models"""
    if model is None:
        return {"message": "No model loaded"}
    
    model_info = {
        "ensemble_architecture": "StackedEnsembleNet",
        "base_models": list(model.base_models.keys()),
        "device": str(device),
        "model_loaded": True
    }
    
    return model_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
