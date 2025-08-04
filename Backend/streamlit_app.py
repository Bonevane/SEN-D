import streamlit as st
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

# Add the current directory to Python path to import architectures
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from architectures import CustomClassifier, FeatureExtractionModel, StackedEnsembleNet

# Page config
st.set_page_config(
    page_title="SEN-D Kidney Stone Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .negative {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .metrics {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
            print(f"Target layer '{self.target_layer_name}' not found. Using last conv layer: {conv_layers[-1][0]}")
            return conv_layers[-1][1]
        
        return None
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Class Activation Map"""
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
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()

def create_heatmap_overlay(original_image, cam, alpha=0.4):
    """Create heatmap overlay on original image"""
    # Print original image dimensions
    if isinstance(original_image, Image.Image):
        print(f"Original PIL image size: {original_image.size}")
        print(f"Original PIL image mode: {original_image.mode}")
    
    # Print CAM dimensions
    print(f"CAM shape: {cam.shape}")
    print(f"CAM dtype: {cam.dtype}")
    
    # Resize CAM to match input image size
    cam_resized = cv2.resize(cam, (299, 299))
    print(f"CAM resized shape: {cam_resized.shape}")
    print("DONE!")
    
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
    
    print(f"Image array shape: {img_array.shape}")
    print(f"Image array dtype: {img_array.dtype}")
    
    print("Creating heatmap overlay...")
    # Create heatmap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap dtype: {heatmap.dtype}")
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
    
    return overlay, heatmap

def get_target_layer_name(model_name):
    """Get appropriate target layer name for different architectures"""
    layer_mapping = {
        'inception_v3': 'backbone.Mixed_7c',
        'inception_resnet_v2': 'backbone.conv2d_7b',
        'xception': 'backbone.conv4'
    }
    return layer_mapping.get(model_name, 'backbone.features')

@st.cache_resource
def load_model(model_type="stacked_ensemble"):
    """Load and cache the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if model_type == "stacked_ensemble":
            model = StackedEnsembleNet(device)
            if os.path.exists('models/stacked_ensemble_meta_learner.pth'):
                model.meta_learner.load_state_dict(
                    torch.load('models/stacked_ensemble_meta_learner.pth', map_location=device)
                )
                st.success("✅ StackedEnsembleNet loaded successfully!")
            else:
                st.warning("⚠️ No saved weights found, using pre-trained backbones only")
        else:
            st.error("Model type not implemented in this demo")
            return None
        
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
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

def predict_all_models(ensemble_model, device, image):
    """Make predictions on all models (base models + ensemble) with Grad-CAM"""
    try:
        input_tensor = preprocess_image(image).to(device)
        class_names = ["Kidney_stone", "Normal"]
        results = {}
        
        start_time = time.time()
        
        # 1. Get predictions from individual base models with Grad-CAM
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
            overlay = None
            heatmap = None
            if cam is not None:
                try:
                    overlay, heatmap = create_heatmap_overlay(image, cam)
                except Exception as e:
                    print(f"Failed to create heatmap for {model_name}: {e}")
            
            results[model_name] = {
                "prediction": class_names[predicted_class],
                "confidence": float(probabilities[predicted_class]),
                "probabilities": {
                    "Kidney_stone": float(probabilities[0]),
                    "Normal": float(probabilities[1])
                },
                "gradcam": cam,
                "overlay": overlay,
                "heatmap": heatmap
            }
            base_predictions.append(probabilities)
        
        # 2. Get ensemble prediction (no Grad-CAM for ensemble as it's a meta-learner)
        ensemble_model.eval()
        with torch.no_grad():
            ensemble_outputs = ensemble_model(input_tensor)
            ensemble_probabilities = torch.softmax(ensemble_outputs, dim=1)[0]
            ensemble_predicted_class = torch.argmax(ensemble_outputs, dim=1).item()
            
            results["ensemble"] = {
                "prediction": class_names[ensemble_predicted_class],
                "confidence": float(ensemble_probabilities[ensemble_predicted_class]),
                "probabilities": {
                    "Kidney_stone": float(ensemble_probabilities[0]),
                    "Normal": float(ensemble_probabilities[1])
                },
                "gradcam": None,  # Ensemble doesn't have meaningful Grad-CAM
                "overlay": None,
                "heatmap": None
            }
        
        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        results["num_models"] = len(ensemble_model.base_models) + 1
        
        return results
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 SEN-D Kidney Stone Detection</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered kidney stone detection from CT scan images**")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a CT scan image (JPG, PNG, JPEG)
        2. Wait for the AI analysis
        3. Review the prediction results
        
        **Supported formats:** JPG, PNG, JPEG
        **Max file size:** 200MB
        """)
        
        st.header("🔧 Model Info")
        st.info("""
        **Architecture:** StackedEnsembleNet
        - InceptionV3
        - InceptionResNetV2  
        - Xception
        - Meta-learner fusion
        
        **Grad-CAM Visualization:**
        - Shows AI focus areas
        - Red/Yellow = High attention
        - Blue/Dark = Low attention
        """)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check model files.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CT scan image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a CT scan image for kidney stone detection"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="CT Scan Image", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            **Filename:** {uploaded_file.name}  
            **Size:** {image.size}  
            **Mode:** {image.mode}
            """)
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image with all models... Please wait."):
                    results = predict_all_models(model, device, image)
                
                if results:
                    # Create tabs for different models
                    tab_names = ["🏆 Ensemble"] + [f"📊 {name.replace('_', ' ').title()}" for name in model.base_models.keys()]
                    tabs = st.tabs(tab_names)
                    
                    # Ensemble results (main tab)
                    with tabs[0]:
                        ensemble_result = results["ensemble"]
                        prediction = ensemble_result["prediction"]
                        confidence = ensemble_result["confidence"]
                        
                        # Color-coded result box
                        if prediction == "Kidney_stone":
                            st.markdown(f'''
                            <div class="prediction-box positive">
                                <h3>🚨 Kidney Stone Detected (Ensemble)</h3>
                                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="prediction-box negative">
                                <h3>✅ Normal - No Kidney Stone (Ensemble)</h3>
                                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Ensemble metrics
                        st.markdown(f'''
                        <div class="metrics">
                            <h4>📊 Ensemble Results</h4>
                            <p><strong>Kidney Stone Probability:</strong> {ensemble_result["probabilities"]["Kidney_stone"]:.2%}</p>
                            <p><strong>Normal Probability:</strong> {ensemble_result["probabilities"]["Normal"]:.2%}</p>
                            <p><strong>Total Processing Time:</strong> {results["processing_time"]:.3f} seconds</p>
                            <p><strong>Models Used:</strong> {results["num_models"]} (3 base + 1 ensemble)</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Progress bars for ensemble
                        st.subheader("📈 Ensemble Probability Breakdown")
                        st.progress(ensemble_result["probabilities"]["Kidney_stone"], 
                                  text=f"Kidney Stone: {ensemble_result['probabilities']['Kidney_stone']:.2%}")
                        st.progress(ensemble_result["probabilities"]["Normal"], 
                                  text=f"Normal: {ensemble_result['probabilities']['Normal']:.2%}")
                    
                    # Individual model results
                    for i, (model_name, base_result) in enumerate([item for item in results.items() if item[0] != "ensemble" and item[0] not in ["processing_time", "num_models"]], 1):
                        with tabs[i]:
                            prediction = base_result["prediction"]
                            confidence = base_result["confidence"]
                            
                            # Create columns for prediction and Grad-CAM
                            pred_col, gradcam_col = st.columns([1, 1])
                            
                            with pred_col:
                                # Model-specific result
                                if prediction == "Kidney_stone":
                                    st.markdown(f'''
                                    <div class="prediction-box positive">
                                        <h3>🚨 Kidney Stone Detected</h3>
                                        <p><strong>Model:</strong> {model_name.replace('_', ' ').title()}</p>
                                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'''
                                    <div class="prediction-box negative">
                                        <h3>✅ Normal - No Kidney Stone</h3>
                                        <p><strong>Model:</strong> {model_name.replace('_', ' ').title()}</p>
                                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                
                                # Individual model progress bars
                                st.subheader(f"📈 {model_name.replace('_', ' ').title()} Probabilities")
                                st.progress(base_result["probabilities"]["Kidney_stone"], 
                                          text=f"Kidney Stone: {base_result['probabilities']['Kidney_stone']:.2%}")
                                st.progress(base_result["probabilities"]["Normal"], 
                                          text=f"Normal: {base_result['probabilities']['Normal']:.2%}")
                            
                            with gradcam_col:
                                # Grad-CAM visualization
                                st.subheader(f"🔍 {model_name.replace('_', ' ').title()} Focus Areas")
                                
                                if base_result["overlay"] is not None:
                                    # Display the overlay
                                    st.image(base_result["overlay"], 
                                           caption=f"Grad-CAM Overlay - {model_name.replace('_', ' ').title()}", 
                                           use_container_width=True)
                                    
                                    # Add explanation
                                    st.info("""
                                    🔍 **Grad-CAM Explanation:**
                                    - **Red/Yellow areas**: High attention (important for prediction)
                                    - **Blue/Dark areas**: Low attention
                                    - Shows which parts of the image the AI focused on
                                    """)
                                    
                                    # Option to show just the heatmap
                                    if st.checkbox(f"Show pure heatmap - {model_name}", key=f"heatmap_{model_name}"):
                                        if base_result["heatmap"] is not None:
                                            st.image(base_result["heatmap"], 
                                                   caption=f"Pure Heatmap - {model_name.replace('_', ' ').title()}", 
                                                   use_container_width=True)
                                else:
                                    st.warning("⚠️ Grad-CAM visualization not available for this model")
                                    st.info("This might be due to model architecture limitations or processing errors.")
                    
                    # Comparison table
                    st.subheader("📋 Model Comparison")
                    comparison_data = []
                    for model_name in ["ensemble"] + list(model.base_models.keys()):
                        result = results[model_name]
                        comparison_data.append({
                            "Model": model_name.replace('_', ' ').title(),
                            "Prediction": result["prediction"],
                            "Confidence": f"{result['confidence']:.2%}",
                            "Kidney Stone Prob": f"{result['probabilities']['Kidney_stone']:.2%}",
                            "Normal Prob": f"{result['probabilities']['Normal']:.2%}"
                        })
                    
                    st.dataframe(comparison_data, use_container_width=True)
                    
                    # Grad-CAM Comparison Section
                    st.subheader("🔍 Grad-CAM Comparison - All Models")
                    st.markdown("Compare how different models focus on different areas of the image:")
                    
                    gradcam_cols = st.columns(3)
                    model_names = list(model.base_models.keys())
                    
                    for i, model_name in enumerate(model_names):
                        with gradcam_cols[i]:
                            base_result = results[model_name]
                            st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                            
                            if base_result["overlay"] is not None:
                                st.image(base_result["overlay"], 
                                       caption=f"{model_name.replace('_', ' ').title()} Focus", 
                                       use_container_width=True)
                                st.markdown(f"**Prediction:** {base_result['prediction']}")
                                st.markdown(f"**Confidence:** {base_result['confidence']:.2%}")
                            else:
                                st.warning("Grad-CAM not available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 SEN-D Kidney Stone Detection System | Built with Streamlit & PyTorch</p>
        <p><em>⚠️ This tool is for research purposes only and should not replace professional medical diagnosis.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
