# SEN-D Kidney Stone Detection FastAPI

This FastAPI application provides a REST API endpoint for kidney stone detection from CT scan images using the StackedEnsembleNet model.

## Features

- **Single endpoint** `/predict` for kidney stone detection
- **Multi-model predictions** (InceptionV3, InceptionResNetV2, Xception + Ensemble)
- **Grad-CAM visualizations** for explainable AI
- **Base64 encoded images** in response for easy integration
- **Health check endpoints** for monitoring
- **Automatic model loading** on startup

## Installation

1. Install dependencies:

```bash
pip install -r fastapi_requirements.txt
```

2. Ensure your model files are in the correct location:

```
Backend/
├── models/
│   └── stacked_ensemble_meta_learner.pth
├── architectures.py
└── fastapi_app.py
```

## Running the API

Start the FastAPI server:

```bash
python fastapi_app.py
```

Or using uvicorn directly:

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Health Check

- **GET** `/` - Basic health check
- **GET** `/health` - Detailed health information
- **GET** `/models` - Model information

### 2. Prediction

- **POST** `/predict` - Upload CT scan image for kidney stone detection

## Usage Examples

### Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@ct_scan.jpg"
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
with open("ct_scan.jpg", "rb") as f:
    files = {"file": ("ct_scan.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()
    print(result)
```

### Using the test script

```bash
python test_api.py
```

## API Response Format

The `/predict` endpoint returns a JSON response with the following structure:

```json
{
  "ensemble": {
    "prediction": "Normal",
    "confidence": 0.85,
    "probabilities": {
      "Kidney_stone": 0.15,
      "Normal": 0.85
    }
  },
  "individual_models": {
    "inception_v3": {
      "prediction": "Normal",
      "confidence": 0.82,
      "probabilities": {
        "Kidney_stone": 0.18,
        "Normal": 0.82
      },
      "gradcam_overlay": "base64_encoded_image",
      "gradcam_heatmap": "base64_encoded_image"
    },
    "inception_resnet_v2": { ... },
    "xception": { ... }
  },
  "processing_time": 2.345,
  "num_models": 4,
  "success": true,
  "message": "Prediction completed successfully"
}
```

## Grad-CAM Visualizations

Each individual model returns:

- `gradcam_overlay`: Original image with heatmap overlay (base64 encoded)
- `gradcam_heatmap`: Pure heatmap visualization (base64 encoded)

To decode and save the images:

```python
import base64
from PIL import Image
import io

# Decode base64 image
image_data = base64.b64decode(result['individual_models']['inception_v3']['gradcam_overlay'])
image = Image.open(io.BytesIO(image_data))
image.save('gradcam_overlay.png')
```

## Interactive API Documentation

Once the server is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Error Handling

The API handles various error scenarios:

- Invalid file types (non-images)
- Model loading failures
- Processing errors
- Network issues

All errors return appropriate HTTP status codes and descriptive messages.

## Performance Notes

- Model is loaded once on startup for better performance
- Grad-CAM generation adds ~1-2 seconds to processing time
- GPU acceleration is automatically used if available
- Consider implementing caching for production use

## Production Deployment

For production deployment, consider:

1. Using a production ASGI server (gunicorn + uvicorn)
2. Adding authentication/authorization
3. Implementing rate limiting
4. Adding logging and monitoring
5. Using container deployment (Docker)
6. Setting up load balancing for multiple instances
