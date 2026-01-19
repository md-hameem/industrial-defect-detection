"""
FastAPI Backend for Industrial Defect Detection

Provides REST API for model inference on uploaded images.
"""

import os
import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from PIL import Image

# Add thesis project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference import ModelInference

# JWT settings (simple auth)
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Simple user database (in-memory for demo)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
    }
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Upload images to detect defects using trained autoencoder models",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model inference
inference = ModelInference()


# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    disabled: Optional[bool] = None


class PredictionResponse(BaseModel):
    success: bool
    model: str
    category: str
    anomaly_score: float
    original_image: str  # base64
    reconstruction: str  # base64
    heatmap: str  # base64
    processing_time: float


class ModelInfo(BaseModel):
    name: str
    type: str
    category: str
    file_size_mb: float


# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user(username: str):
    if username in USERS_DB:
        return USERS_DB[username]
    return None


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user_optional(token: str = Depends(oauth2_scheme)):
    """Optional authentication - returns None if no token"""
    if token is None:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        user = get_user(username)
        return user
    except JWTError:
        return None


# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Industrial Defect Detection API",
        "version": "1.0.0",
    }


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available trained models"""
    return inference.get_available_models()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = "CAE",
    category: str = "bottle",
    user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Upload an image and get defect detection results.
    
    - **file**: Image file (PNG, JPG)
    - **model_type**: CAE, VAE, DAE, or CNN
    - **category**: MVTec category (not needed for CNN)
    """
    import time
    start_time = time.time()
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Run inference
    try:
        result = inference.predict(image, model_type, category)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    processing_time = time.time() - start_time
    
    # Return different response based on model type
    if result.get("is_classifier", False):
        # CNN classifier response
        return {
            "success": True,
            "model": "CNN",
            "model_type": "classifier",
            "category": "NEU",
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "class_probabilities": result["class_probabilities"],
            "original_image": result["original_base64"],
            "chart_image": result["chart_base64"],
            "processing_time": processing_time,
        }
    else:
        # Autoencoder response
        return {
            "success": True,
            "model": model_type,
            "model_type": "autoencoder",
            "category": category,
            "anomaly_score": result["anomaly_score"],
            "original_image": result["original_base64"],
            "reconstruction": result["reconstruction_base64"],
            "heatmap": result["heatmap_base64"],
            "processing_time": processing_time,
        }


@app.get("/categories")
async def get_categories():
    """Get list of available categories"""
    return inference.get_available_categories()


@app.get("/cnn/available")
async def cnn_available():
    """Check if CNN classifier model is available"""
    return {"available": inference.is_cnn_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

