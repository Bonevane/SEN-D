from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import io
import os
import sys

# Ensure the current directory is in the system path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app_utils import (
    PredictionResponse,
    load_model,
    predict_all_models
)

# Global variables for model caching
model = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, device
    model, device = load_model()
    if model is None:
        print("Failed to load model on startup!")
    yield
    print("Shutting down app...")

app = FastAPI(
    title="SEN-D Kidney Stone Detection API",
    description="AI-powered kidney stone detection from CT scan images",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "https://sen-d.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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