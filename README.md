# SEN-D: AI-Powered Kidney Stone Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)

> **SEN-D** (StackedEnsembleNet Detection) is an advanced AI system for automated kidney stone detection from CT scan images using deep learning ensemble methods with explainable AI visualizations.

## 🩺 Overview

SEN-D uses a sophisticated ensemble of convolutional neural networks to provide accurate, reliable kidney stone detection with **98.74% accuracy**. The system combines multiple CNN architectures with Grad-CAM visualizations to assist radiologists in clinical diagnosis.

### Key Features

- 🎯 **High Accuracy**: 98.74% classification accuracy with 98.57% precision
- 🧠 **Ensemble Learning**: Combines InceptionV3, InceptionResNetV2, and Xception models
- 🔍 **Explainable AI**: Grad-CAM visualizations for transparent decision making
- 🚀 **Production Ready**: FastAPI backend deployed on Fly.io with React frontend
- 📊 **Real-time Analysis**: Fast prediction with comprehensive model outputs
- 🏥 **Clinical Grade**: Validated on hospital-grade CT imaging data

## 🏗️ Architecture

### Model Architecture

```
          Input CT Scan (299x299)
                    ↓
┌─────────────┬─────────────-┬─────────────┐
│ InceptionV3 │InceptionResV2│  Xception   │
└─────────────┴─────────────-┴─────────────┘
     ↓               ↓               ↓
┌─────────────┬─────────────-┬─────────────┐
│ Custom Head │ Custom Head  │ Custom Head │
│  256→128→2  │  256→128→2   │  256→128→2  │
└─────────────┴─────────────-┴─────────────┘
     ↓               ↓               ↓
      StackedEnsembleNet Meta-Learner
                    ↓
          Final Classification
         (Kidney Stone / Normal)
```

### Technology Stack

**Frontend:**

- React 19 with Vite
- TailwindCSS for styling
- React Icons for UI components
- Responsive design with animations
- Deployed on Vercel

**Backend:**

- FastAPI for REST API
- PyTorch for deep learning
- Grad-CAM for explainability
- Docker containerization
- Fly.io deployment

**AI/ML:**

- Custom ensemble architecture (See References)
- Transfer learning with pre-trained models
- Data augmentation pipeline
- Cross-entropy loss optimization

## 📊 Performance Metrics

| Metric                 | Score  |
| ---------------------- | ------ |
| **Accuracy**           | 98.74% |
| **Precision**          | 98.57% |
| **Recall/Sensitivity** | 98.96% |
| **F1-Score**           | 98.76% |
| **MCC**                | 97.48% |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- GPU recommended for training (CUDA support)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/SEN-D.git
cd SEN-D
```

2. **Backend Setup**

```bash
cd Backend
pip install -r requirements.txt
```

3. **Frontend Setup**

```bash
cd Frontend
npm install
```

### Running the Application

1. **Start the Backend**

```bash
cd Backend
python app.py
# API available at http://localhost:8000
```

2. **Start the Frontend**

```bash
cd Frontend
npm run dev
# Frontend available at http://localhost:5173
```

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@ct_scan.jpg"
```

### Python Example

```python
import requests

with open("ct_scan.jpg", "rb") as f:
    files = {"file": ("ct_scan.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()

print(f"Prediction: {result['ensemble']['prediction']}")
print(f"Confidence: {result['ensemble']['confidence']:.2%}")
```

### Response Format

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
      "probabilities": {...},
      "gradcam_overlay": "base64_encoded_image",
      "gradcam_heatmap": "base64_encoded_image"
    },
    "inception_resnet_v2": {...},
    "xception": {...}
  },
  "processing_time": 2.345,
  "success": true
}
```

## 🧪 Training Your Own Model

### Dataset Preparation

The project uses a hybrid dataset from:

1. [Kaggle Axial CT Imaging Dataset](https://www.kaggle.com/datasets/orvile/axial-ct-imaging-dataset-kidney-stone-detection)
2. Elazığ Fethi Sekin City Hospital, Turkey ([Reference](https://www.sciencedirect.com/science/article/abs/pii/S0010482521003632))

Place your dataset in:

```
Backend/Dataset/
├── Train/
│   ├── Kidney_stone/
│   └── Normal/
└── Test/
    ├── Kidney_stone/
    └── Normal/
```

### Training Process

1. **Data Augmentation**

```bash
cd Backend
python augment.py  # Generate augmented training data
```

2. **Open Training Notebook**

```bash
jupyter notebook Notebook.ipynb
```

3. **Follow the notebook sections:**
   - Data loading and preprocessing
   - Model architecture setup
   - Training individual models
   - Ensemble training
   - Evaluation and testing

### Model Configuration

The training supports multiple architectures:

- `inception_v3`: InceptionV3 with custom classifier
- `inception_resnet_v2`: InceptionResNetV2 with residual connections
- `xception`: Xception with depthwise separable convolutions
- `stacked_ensemble`: Complete ensemble with meta-learner

## 📁 Project Structure

```
SEN-D/
├── Frontend/                 # React application
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── assets/          # Static assets
│   │   └── main.jsx         # Application entry
│   ├── public/              # Public assets
│   ├── package.json
│   └── vite.config.ts
├── Backend/                 # FastAPI application
│   ├── Dataset/            # Training/test data
│   │   ├── Train/
│   │   └── Test/
│   ├── Dataset_Augmented/  # Augmented training data
│   ├── models/             # Trained model files
│   ├── app.py              # FastAPI main application
│   ├── app_utils.py        # Utility functions
│   ├── architectures.py    # Model architectures
│   ├── augment.py          # Data augmentation
│   ├── Notebook.ipynb      # Training notebook
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Container configuration
├── README.md
└── LICENSE
```

## 🔬 Technical Requirements

### Image Guidelines

**Good Examples:**

- Single CT scan slices (299x299+ pixels)
- Clear kidney regions visible
- Proper contrast and orientation
- DICOM windowing applied

**Avoid:**

- Multi-panel reports
- Images with annotations/text
- Non-medical images
- Low resolution scans

### System Requirements

- **Input Size**: Minimum 299x299 pixels
- **Format**: RGB color channels
- **File Size**: Maximum 10MB
- **Supported Types**: JPG, PNG, JPEG

## 🏥 Clinical Validation

This system is based on the research:

> **"An Optimized Fusion of Deep Learning Models for Kidney Stone Detection from CT Images"**  
> _Computers in Biology and Medicine, 2024_ - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824002192)

### Clinical Objective

- Binary classification: Kidney Stone vs Normal
- Support early detection and diagnosis
- Assist radiologists in CT scan analysis
- Improve diagnostic efficiency and accuracy

## 🚢 Deployment

### Fly.io Deployment

```bash
cd Backend
fly deploy
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Any contributions will be greatly appreciated!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Research & References

- [Original Research Paper](https://www.sciencedirect.com/science/article/pii/S1319157824002192)
- [Kaggle Dataset](https://www.kaggle.com/datasets/orvile/axial-ct-imaging-dataset-kidney-stone-detection)
- [Hospital Dataset Reference](https://www.sciencedirect.com/science/article/abs/pii/S0010482521003632)

## 🙏 Acknowledgments

- Research team for the original StackedEnsembleNet architecture
- Kaggle community for the CT imaging dataset
- Elazığ Fethi Sekin City Hospital for clinical data
- Open source community for the amazing tools and libraries

---

**⚠️ Disclaimer**: This system is for research and educational purposes. Always consult qualified medical professionals for clinical diagnosis and treatment decisions.
