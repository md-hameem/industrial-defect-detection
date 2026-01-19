# Web Application Backend

FastAPI backend for Industrial Defect Detection inference.

## Setup

```bash
cd web/backend
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

- `GET /` - Health check
- `GET /models` - List available models
- `POST /predict` - Upload image and get prediction
- `POST /login` - Simple authentication
