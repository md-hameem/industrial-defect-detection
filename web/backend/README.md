# Web Application Backend

FastAPI backend for Industrial Defect Detection inference.

## Features

- 🔍 **Multi-Model Inference** - CAE, VAE, DAE (autoencoders) + CNN (classifier)
- 🌡️ **Heatmap Generation** - Visual anomaly localization
- 📊 **Class Probabilities** - CNN bar chart visualization
- 🔐 **JWT Authentication** - Simple token-based auth
- ⚡ **Fast Response** - Optimized model caching

## Setup

```bash
cd web/backend
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

API available at http://localhost:8000

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/models` | List available trained models |
| `GET` | `/categories` | List MVTec categories |
| `GET` | `/cnn/available` | Check if CNN model exists |
| `POST` | `/predict` | Upload image and get prediction |
| `POST` | `/login` | Get JWT token |

## Prediction API

### Request
```bash
POST /predict?model_type=CAE&category=bottle
Content-Type: multipart/form-data
file: <image.png>
```

### Autoencoder Response
```json
{
  "success": true,
  "model": "CAE",
  "model_type": "autoencoder",
  "category": "bottle",
  "anomaly_score": 0.234,
  "original_image": "base64...",
  "reconstruction": "base64...",
  "heatmap": "base64...",
  "processing_time": 1.23
}
```

### CNN Classifier Response
```json
{
  "success": true,
  "model": "CNN",
  "model_type": "classifier",
  "category": "NEU",
  "predicted_class": "Scratches",
  "confidence": 0.987,
  "class_probabilities": {
    "Crazing": 0.001,
    "Inclusion": 0.002,
    "Patches": 0.003,
    "Pitted": 0.002,
    "Rolled": 0.005,
    "Scratches": 0.987
  },
  "original_image": "base64...",
  "chart_image": "base64...",
  "processing_time": 0.45
}
```

## Model Files

Models are loaded from `../../outputs/models/`:
- `cae_{category}_final.pth` - Convolutional Autoencoder
- `vae_{category}_final.pth` - Variational Autoencoder
- `dae_{category}_final.pth` - Denoising Autoencoder
- `cnn_classifier_final.pth` - CNN Classifier (NEU 6-class)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT secret key | `secret-key-change-me` |

## Dependencies

- FastAPI
- Uvicorn
- PyTorch
- Pillow
- NumPy
- Matplotlib
- python-jose (JWT)
