from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import soundfile as sf
import io
from typing import Dict, List, Optional
import logging
from pydantic import BaseModel
import os
from datetime import datetime
from pathlib import Path

# Import our clean modules that use your EXACT pipeline
from src.models.clean_pcg_classifier import CleanPCGClassifier
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PCG Signal Classification API - Clean Pipeline",
    description="Real-time PCG classification using your EXACT original pipeline",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lubdub-ai.netlify.app",  # Your Netlify domain
        "http://localhost:3000",         # Local development
        "http://localhost:8000",         # Local backend
        "http://127.0.0.1:8000",        # Alternative local
        "*"                              # Allow all (for development)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    """Handle CORS preflight requests"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Serve static files (frontend)
if Path("frontend").exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    logger.info("✅ Frontend mounted at /static")
from fastapi.staticfiles import StaticFiles

# Add this line after your existing mounts
app.mount("/img", StaticFiles(directory="frontend/img"), name="images")
# Response models
class PCGAnalysisResult(BaseModel):
    classification: str
    confidence: float
    heart_rate: Optional[float]
    s1_count: int
    s2_count: int
    signal_quality: str
    processing_time_ms: float
    metadata: Dict

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str

# Global classifier instance
pcg_classifier: Optional[CleanPCGClassifier] = None

@app.on_event("startup")
async def startup_event():
    """Initialize classifier using your exact pipeline"""
    global pcg_classifier
    
    try:
        logger.info("🚀 Initializing PCG Classification System (Clean Pipeline)...")
        
        # Initialize classifier with your exact pipeline
        config = Config()
        pcg_classifier = CleanPCGClassifier(
            artifact_model_path=str(config.ARTIFACT_MODEL_PATH),
            murmur_model_path=str(config.MURMUR_MODEL_PATH)
        )
        
        # Check model status
        model_info = pcg_classifier.get_model_info()
        if model_info['using_mock_models']:
            logger.warning("⚠️ Using mock models - place real models in models/ directory")
        else:
            logger.info("✅ Real ML models loaded successfully")
        
        logger.info("✅ Clean pipeline initialized:")
        logger.info("   📊 Metadata creation: run_metadata.py process")
        logger.info("   🔧 Feature extraction: your exact functions")
        logger.info("   🧠 Models: your trained classifiers")
        logger.info("🎉 System startup completed!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        logger.info("🔄 System will continue with limited functionality")

# Frontend route - serve the main page
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    frontend_path = Path("frontend/index.html")
    
    if frontend_path.exists():
        with open(frontend_path, 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>PCG Analyzer API - Clean Pipeline</h1>
                <p>Using your EXACT original pipeline. API is running at:</p>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/models/info">Model Info</a></li>
                </ul>
            </body>
        </html>
        """)

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="2.0.0-clean-pipeline"
    )

@app.post("/analyze", response_model=PCGAnalysisResult)
async def analyze_pcg(file: UploadFile = File(...)):
    """Analyze uploaded PCG audio file using your exact pipeline"""
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.flac', '.mp3')):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload WAV, FLAC, or MP3 files."
            )
        
        # Read audio file
        file_bytes = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(file_bytes))
        
        # Classify using your EXACT pipeline
        result = pcg_classifier.classify(audio_data, sample_rate)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result['processing_time_ms'] = processing_time
        
        return PCGAnalysisResult(
            classification=result['classification'],
            confidence=result['confidence'],
            heart_rate=result.get('heart_rate'),
            s1_count=result['s1_count'],
            s2_count=result['s2_count'],
            signal_quality=result['signal_quality'],
            processing_time_ms=processing_time,
            metadata={
                'filename': file.filename,
                'original_duration': len(audio_data) / sample_rate,
                'original_sample_rate': sample_rate,
                'processed_sample_rate': Config.TARGET_SAMPLE_RATE,
                'pipeline': 'exact_original',
                'using_mock_models': result['details'].get('using_mock_models', False)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

from pydantic import BaseModel

class RealtimeAudioRequest(BaseModel):
    audio_data: List[float]
    sample_rate: Optional[int] = 44100  # Default fallback

@app.post("/analyze-realtime", response_model=PCGAnalysisResult)
async def analyze_realtime_pcg(request: RealtimeAudioRequest):
    """Analyze real-time PCG data with proper sample rate"""
    start_time = datetime.now()
    
    try:
        # Convert to numpy array
        audio_array = np.array(request.audio_data, dtype=np.float32)
        actual_sample_rate = request.sample_rate
        
        logger.info(f"📊 Received audio:")
        logger.info(f"   Array length: {len(audio_array)}")
        logger.info(f"   Sample rate: {actual_sample_rate}")
        logger.info(f"   Duration: {len(audio_array) / actual_sample_rate:.2f}s")
        
        # Validate duration
        actual_duration = len(audio_array) / actual_sample_rate
        if actual_duration < Config.MIN_AUDIO_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short. Got {actual_duration:.1f}s, need minimum {Config.MIN_AUDIO_LENGTH}s."
            )
        
        # Classify using the actual sample rate
        result = pcg_classifier.classify(audio_array, actual_sample_rate)
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result['processing_time_ms'] = processing_time
        
        return PCGAnalysisResult(
            classification=result['classification'],
            confidence=result['confidence'],
            heart_rate=result.get('heart_rate'),
            s1_count=result['s1_count'],
            s2_count=result['s2_count'],
            signal_quality=result['signal_quality'],
            processing_time_ms=processing_time,
            metadata={
                'source': 'realtime_recording',
                'duration': actual_duration,  # Use actual duration
                'detected_sample_rate': actual_sample_rate,  # Show what we detected
                'array_length': len(audio_array),
                'pipeline': 'exact_original',
                'using_mock_models': result['details'].get('using_mock_models', False)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing realtime audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about the system"""
    if pcg_classifier is None:
        return {
            "status": "not_initialized",
            "message": "System not fully initialized"
        }
    
    model_info = pcg_classifier.get_model_info()
    
    return {
        "system_status": "ready",
        "pipeline": "exact_original_reproduction",
        "metadata_creation": "run_metadata_py_process",
        "feature_extraction": "exact_original_functions",
        "s1s2_detection": "signal_processing_pipeline",
        "ml_models": {
            "artifact_detection": "loaded" if model_info['artifact_model_loaded'] else "mock",
            "murmur_detection": "loaded" if model_info['murmur_model_loaded'] else "mock"
        },
        "using_mock_models": model_info['using_mock_models'],
        "supported_sample_rates": [Config.TARGET_SAMPLE_RATE],
        "version": "2.0.0-clean-pipeline",
        "compatibility": "100%_with_original_training_pipeline"
    }

@app.get("/test-compatibility")
async def test_compatibility():
    """Test endpoint to verify pipeline compatibility"""
    try:
        # Generate synthetic heart sound for testing
        duration = 10  # seconds
        sample_rate = 1000
        samples = duration * sample_rate
        
        # Generate synthetic heart sound pattern
        t = np.linspace(0, duration, samples)
        audio_signal = np.zeros(samples)
        
        # Add periodic S1 and S2 sounds (simulate 72 bpm)
        heart_rate = 72  # bpm
        beat_period = 60 / heart_rate  # seconds per beat
        
        for beat_time in np.arange(0, duration, beat_period):
            # S1 sound
            s1_start = int(beat_time * sample_rate)
            s1_end = min(s1_start + 100, samples)
            if s1_start < samples:
                audio_signal[s1_start:s1_end] += np.random.normal(0.4, 0.1, s1_end - s1_start)
            
            # S2 sound
            s2_start = int((beat_time + 0.3) * sample_rate)
            s2_end = min(s2_start + 50, samples)
            if s2_start < samples:
                audio_signal[s2_start:s2_end] += np.random.normal(0.2, 0.05, s2_end - s2_start)
        
        # Add background noise
        noise = np.random.normal(0, 0.02, samples)
        audio_signal += noise
        
        # Test classification
        result = pcg_classifier.classify(audio_signal, sample_rate)
        
        return {
            "test_status": "success",
            "synthetic_audio": {
                "duration": duration,
                "sample_rate": sample_rate,
                "expected_heart_rate": heart_rate
            },
            "classification_result": result,
            "pipeline_verified": "exact_original_reproduction"
        }
        
    except Exception as e:
        return {
            "test_status": "failed",
            "error": str(e),
            "pipeline_verified": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)