import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import os
import cv2
import matplotlib.cm as cm
import base64
import io
import gc
from typing import Dict
from pydantic import BaseModel

from architectures import StackedEnsembleNet

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
            self.activations = output.detach()  # Detach to prevent gradient accumulation
        
        def backward_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0].detach()  # Detach to prevent gradient accumulation
        
        # Find the target layer
        target_layer = self._find_target_layer()
        if target_layer is not None:
            self.hooks.append(target_layer.register_forward_hook(forward_hook))
            self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
    
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
            self.model.zero_grad()  # Clear gradients first
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            # Backward pass
            target = output[0, class_idx]
            target.backward(retain_graph=False)  # Don't retain graph
            
            if self.gradients is None or self.activations is None:
                return None
            
            # Generate CAM
            gradients = self.gradients[0]  # Remove batch dimension
            activations = self.activations[0]  # Remove batch dimension
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=[1, 2])
            
            # Weighted combination of activation maps
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # Apply ReLU
            cam = torch.relu(cam)
            
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
            
            cam_numpy = cam.detach().cpu().numpy()
            
            # Clear intermediate tensors
            del gradients, activations, weights, cam, output, target
            
            return cam_numpy
            
        except Exception as e:
            print(f"Error in generate_cam: {e}")
            return None
        finally:
            # Always clear gradients
            self.model.zero_grad()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Clear stored gradients and activations
        self.gradients = None
        self.activations = None

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
            rgb_image.close()  # Close PIL image to free memory
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
        
        # Clean up intermediate arrays
        del cam_resized, img_array, heatmap
        
        return overlay, None  # Don't return heatmap separately to save memory
        
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
    
    image_rgb = image.convert('RGB')
    tensor = transform(image_rgb).unsqueeze(0)
    
    # Close the converted image to free memory
    if image_rgb != image:
        image_rgb.close()
    
    return tensor

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    try:
        # Convert to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_array)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG', optimize=True)  # Add optimize=True
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Clean up
        buffer.close()
        pil_image.close()
        del image_array  # Delete the array
        
        return image_base64
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def predict_all_models(ensemble_model, device, image):
    """Make predictions on all models (base models + ensemble) with Grad-CAM"""
    gradcam_objects = []  # Keep track of GradCAM objects to clean up
    
    try:
        input_tensor = preprocess_image(image).to(device)
        class_names = ["Kidney_stone", "Normal"]
        
        start_time = time.time()
        
        # 1. Get predictions from individual base models with Grad-CAM
        individual_results = {}
        
        for model_name, base_model in ensemble_model.base_models.items():
            base_model.eval()
            
            # Prediction without gradients first
            with torch.no_grad():
                outputs = base_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Clear the outputs tensor
            del outputs
            
            # Generate Grad-CAM for this model (separate from prediction)
            target_layer = get_target_layer_name(model_name)
            gradcam = GradCAM(base_model, target_layer)
            gradcam_objects.append(gradcam)  # Track for cleanup
            
            # Enable gradients for Grad-CAM
            input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
            cam = gradcam.generate_cam(input_tensor_grad, predicted_class)
            
            # Clear the gradient tensor immediately
            del input_tensor_grad
            
            # Create heatmap overlay
            overlay_base64 = None
            if cam is not None:
                try:
                    overlay, _ = create_heatmap_overlay(image, cam)
                    if overlay is not None:
                        overlay_base64 = image_to_base64(overlay)
                        del overlay  # Free overlay immediately
                    del cam  # Free cam array
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
                "gradcam_heatmap": None  # Remove separate heatmap to save memory
            }
            
            # Clear probabilities tensor
            del probabilities
        
        # 2. Get ensemble prediction
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
            
            # Clear ensemble tensors
            del ensemble_outputs, ensemble_probabilities
        
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
    finally:
        # CRITICAL: Clean up all GradCAM objects and their hooks
        for gradcam in gradcam_objects:
            gradcam.remove_hooks()
        gradcam_objects.clear()
        
        # Clear input tensor
        if 'input_tensor' in locals():
            del input_tensor
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()