from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import io
from typing import Dict, List, Optional
import logging
from pydantic import BaseModel
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PCG Signal Classification API - Testing Mode",
    description="Real-time PCG signal classification (Testing without ML models)",
    version="1.0.0-testing"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    mode: str

# Mock classifier for testing
class MockPCGClassifier:
    """Mock classifier that simulates ML predictions without requiring actual models"""
    
    def __init__(self):
        self.loaded = True
        logger.info("Mock PCG Classifier initialized")
    
    def classify_audio_array(self, audio_data: np.ndarray) -> Dict:
        """Mock classification that analyzes basic signal properties"""
        
        # Basic signal analysis
        duration = len(audio_data) / 1000.0  # Assuming 1000 Hz
        signal_power = np.mean(audio_data ** 2)
        max_amplitude = np.max(np.abs(audio_data))
        std_amplitude = np.std(audio_data)
        
        # Mock heart rate estimation (based on signal length)
        estimated_beats = duration * 1.2  # Assume ~72 bpm
        heart_rate = (estimated_beats / duration) * 60
        
        # Mock S1/S2 detection (based on duration)
        s1_count = int(estimated_beats)
        s2_count = int(estimated_beats)
        
        # Mock classification logic
        if signal_power < 0.001:
            classification = "artifact"
            confidence = 0.9
            signal_quality = "poor"
        elif max_amplitude > 0.8:
            classification = "artifact" 
            confidence = 0.85
            signal_quality = "poor"
        elif std_amplitude > 0.1:
            classification = "murmur"
            confidence = 0.75
            signal_quality = "good"
        else:
            classification = "healthy"
            confidence = 0.8
            signal_quality = "good"
        
        return {
            "classification": classification,
            "confidence": confidence,
            "heart_rate": heart_rate,
            "s1_count": s1_count,
            "s2_count": s2_count,
            "signal_quality": signal_quality
        }

# Global classifier instance
mock_classifier = MockPCGClassifier()

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0-testing",
        mode="testing_without_ml_models"
    )

@app.post("/analyze", response_model=PCGAnalysisResult)
async def analyze_pcg(file: UploadFile = File(...)):
    """
    Analyze uploaded PCG audio file (mock version)
    """
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
        
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(file_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read audio file: {str(e)}")
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 1000 Hz (simple decimation for testing)
        if sample_rate != 1000:
            decimation_factor = int(sample_rate / 1000)
            if decimation_factor > 1:
                audio_data = audio_data[::decimation_factor]
        
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Mock classification
        result = mock_classifier.classify_audio_array(audio_data)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
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
                'original_duration': len(audio_data) / 1000.0,
                'original_sample_rate': sample_rate,
                'mode': 'mock_classification'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze-realtime", response_model=PCGAnalysisResult)
async def analyze_realtime_pcg(audio_data: List[float]):
    """
    Analyze real-time PCG data from web recording (mock version)
    """
    start_time = datetime.now()
    
    try:
        # Convert to numpy array
        audio_array = np.array(audio_data, dtype=np.float32)
        
        # Validate audio length (minimum 5 seconds for meaningful analysis)
        if len(audio_array) < 5000:  # 5 seconds at 1000 Hz
            raise HTTPException(
                status_code=400,
                detail="Audio too short. Minimum 5 seconds required for analysis."
            )
        
        # Normalize
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Mock classification
        result = mock_classifier.classify_audio_array(audio_array)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
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
                'duration': len(audio_array) / 1000.0,
                'sample_rate': 1000,
                'mode': 'mock_classification'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing realtime audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "mode": "testing",
        "message": "Running in testing mode with mock models",
        "mock_classifier": "active",
        "real_models": "not_loaded",
        "note": "This version simulates ML predictions for testing purposes"
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "API is working!",
        "numpy_version": np.__version__,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)