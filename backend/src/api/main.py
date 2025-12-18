import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import pyarrow.feather as feather
from config import (
    API_NAME, API_VERSION, API_HOST, API_PORT, 
    CORS_ORIGINS, DEBUG_MODE, FEATURE_ENGINEERED_FEATHER
)
from utils.logger import setup_logging, APILoggingMiddleware
from core.model import ModelManager
from api.routers import health, prediction

# 1. Setup Logging
logger = setup_logging()

# 2. Lifecycle Manager (Startup/Shutdown events)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Starting up Sales Forecasting API...")
    
    # Initialize Model Manager and Load Models
    model_manager = ModelManager()
    success = model_manager.load_models()
    
    if success:
        logger.info(f"✅ Models loaded successfully. Available keys: {len(model_manager.models_dict) if model_manager.models_dict else 0}")
    else:
        logger.error("❌ Failed to load models on startup. API will assume no models available.")
        
    # Store in app state for global access
    app.state.model_manager = model_manager
    
    # Load Feature Data (For prediction context)
    logger.info("Loading feature engineered data...")
    try:
        # Check if file exists first to avoid confusing arrow errors
        if FEATURE_ENGINEERED_FEATHER.exists():
            feature_data = feather.read_feather(FEATURE_ENGINEERED_FEATHER)
            app.state.feature_data = feature_data
            logger.info(f"✅ Feature data loaded successfully. Shape: {feature_data.shape}")
        else:
            logger.error(f"❌ Feature file not found at: {FEATURE_ENGINEERED_FEATHER}")
            app.state.feature_data = None
    except Exception as e:
        logger.error(f"❌ Failed to load feature data: {str(e)}")
        app.state.feature_data = None
    
    yield
    
    # --- Shutdown ---
    logger.info("Shutting down API...")

# 3. Create FastAPI App
app = FastAPI(
    title=API_NAME,
    version=API_VERSION,
    description="API for Sales Forecasting with Explainable AI (XAI)",
    lifespan=lifespan,
    debug=DEBUG_MODE
)

# 4. Add Middlewares

# CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Logging Middleware (Custom security & performance logging)
app.add_middleware(APILoggingMiddleware)

# 5. Add Timing Header (Optional but useful)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 6. Include Routers
app.include_router(health.router)
app.include_router(prediction.router)

# 7. Root Endpoint
@app.get("/", tags=["General"])
async def root():
    return {
        "message": f"Welcome to {API_NAME}",
        "version": API_VERSION,
        "docs_url": "/docs",
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Running server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api.main:app", 
        host=API_HOST, 
        port=API_PORT, 
        reload=DEBUG_MODE
    )
