import requests
import json
import base64
from PIL import Image
import io

def test_api():
    """Test the FastAPI kidney stone detection endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Test image path (replace with your actual image path)
    image_path = "test_image.jpg"
    
    try:
        # Read and send image
        with open(image_path, "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            print("🎉 Prediction successful!")
            print(f"Processing time: {result['processing_time']:.3f} seconds")
            print(f"Models used: {result['num_models']}")
            print()
            
            # Ensemble results
            ensemble = result['ensemble']
            print("🏆 ENSEMBLE PREDICTION:")
            print(f"  Prediction: {ensemble['prediction']}")
            print(f"  Confidence: {ensemble['confidence']:.2%}")
            print(f"  Kidney Stone Prob: {ensemble['probabilities']['Kidney_stone']:.2%}")
            print(f"  Normal Prob: {ensemble['probabilities']['Normal']:.2%}")
            print()
            
            # Individual model results
            print("📊 INDIVIDUAL MODEL PREDICTIONS:")
            for model_name, model_result in result['individual_models'].items():
                print(f"  {model_name.replace('_', ' ').title()}:")
                print(f"    Prediction: {model_result['prediction']}")
                print(f"    Confidence: {model_result['confidence']:.2%}")
                print(f"    Grad-CAM available: {'Yes' if model_result['gradcam_overlay'] else 'No'}")
                print()
                
                # Save Grad-CAM visualizations if available
                if model_result['gradcam_overlay']:
                    overlay_data = base64.b64decode(model_result['gradcam_overlay'])
                    overlay_image = Image.open(io.BytesIO(overlay_data))
                    overlay_image.save(f"gradcam_overlay_{model_name}.png")
                    print(f"    Saved Grad-CAM overlay: gradcam_overlay_{model_name}.png")
                
                if model_result['gradcam_heatmap']:
                    heatmap_data = base64.b64decode(model_result['gradcam_heatmap'])
                    heatmap_image = Image.open(io.BytesIO(heatmap_data))
                    heatmap_image.save(f"gradcam_heatmap_{model_name}.png")
                    print(f"    Saved Grad-CAM heatmap: gradcam_heatmap_{model_name}.png")
        
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        print("Please update the image_path variable with a valid image file.")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the FastAPI server is running.")
        print("Run: python fastapi_app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            result = response.json()
            print("🔧 API Health Check:")
            print(f"  Status: {result['status']}")
            print(f"  Model loaded: {result['model_loaded']}")
            print(f"  Device: {result['device']}")
            print(f"  Base models: {result['num_base_models']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🩺 SEN-D Kidney Stone Detection API Test")
    print("=" * 50)
    
    # Test health endpoint
    test_health()
    print()
    
    # Test prediction endpoint
    test_api()
