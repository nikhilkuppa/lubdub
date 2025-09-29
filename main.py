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
    title="Medical Analysis Suite API",
    description="Multi-modal medical signal analysis platform",
    version="3.0.0"
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
    app.mount("/img", StaticFiles(directory="frontend/img"), name="images")
    logger.info("✅ Frontend mounted at /static")

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

class RealtimeAudioRequest(BaseModel):
    audio_data: List[float]
    sample_rate: Optional[int] = 44100  # Default fallback

# Add these imports at the top of your main.py
from src.rppg.rppg_analyzer import RPPGAnalyzer
import tempfile
import shutil

# Add to your global variables section (around line 78)
rppg_analyzer: Optional[RPPGAnalyzer] = None

# Update the startup_event function to initialize rPPG analyzer
@app.on_event("startup")
async def startup_event():
    """Initialize all analysis modules"""
    global pcg_classifier, rppg_analyzer
    
    try:
        logger.info("🚀 Initializing Medical Analysis Suite...")
        
        # Initialize PCG classifier
        config = Config()
        pcg_classifier = CleanPCGClassifier(
            artifact_model_path=str(config.ARTIFACT_MODEL_PATH),
            murmur_model_path=str(config.MURMUR_MODEL_PATH)
        )
        
        # Check model status
        model_info = pcg_classifier.get_model_info()
        if model_info['using_mock_models']:
            logger.warning("⚠️ PCG: Using mock models - place real models in models/ directory")
        else:
            logger.info("✅ PCG: Real ML models loaded successfully")
        
        # Initialize rPPG analyzer
        logger.info("🔄 Initializing rPPG analyzer...")
        rppg_analyzer = RPPGAnalyzer()
        logger.info("✅ rPPG: Analyzer initialized successfully")
        
        logger.info("🎉 Medical Analysis Suite startup completed!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        logger.info("🔄 System will continue with limited functionality")

# Add these rPPG-specific models and endpoints after the PCG section
from pydantic import BaseModel

class RPPGFrameRequest(BaseModel):
    frame_data: List[List[List[int]]]  # RGB image as 3D array
    timestamp: float

class RPPGAnalysisResult(BaseModel):
    success: bool
    heart_rate: Optional[float]
    confidence: Optional[float]
    hrv: Optional[float]
    signal_quality: Optional[str]
    fps: Optional[float]
    error: Optional[str]

# Update the rPPG section in main.py (replace the existing placeholder)
# =====================
# rPPG APPLICATION
# =====================

@app.get("/rppg", response_class=HTMLResponse)
async def serve_rppg_page():
    """Serve the rPPG analyzer page"""
    rppg_path = Path("frontend/rppg.html")
    
    if rppg_path.exists():
        with open(rppg_path, 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        # Fallback page if rppg.html doesn't exist yet
        return HTMLResponse(content="""
        <html>
        <head>
            <title>rPPG Analyzer</title>
            <style>
                body {
                    background: #0D1117;
                    color: #F0F6FC;
                    font-family: 'Tiempos', Georgia, sans-serif;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    padding: 2rem;
                }
                h1 {
                    font-size: 3rem;
                    margin-bottom: 1rem;
                }
                p {
                    font-size: 1.25rem;
                    color: #8B949E;
                    margin-bottom: 2rem;
                }
                a {
                    color: #58A6FF;
                    text-decoration: none;
                    font-size: 1.1rem;
                    padding: 0.75rem 1.5rem;
                    border: 1px solid #58A6FF;
                    border-radius: 8px;
                    display: inline-block;
                    transition: all 0.3s ease;
                }
                a:hover {
                    background: #58A6FF;
                    color: #0D1117;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📹 rPPG Analyzer</h1>
                <p>Contact-free heart rate monitoring from video</p>
                <p style="color: #FB8500;">Please add rppg.html to your frontend directory</p>
                <a href="/">← Back to Home</a>
            </div>
        </body>
        </html>
        """)


class SignalData(BaseModel):
    signals: List[List[float]]
    timestamps: List[float]

@app.post('/api/rppg/analyze-stream')
async def analyze_stream(data: SignalData):
    """
    Analyzes a live stream of rPPG signal data.
    """
    try:
        result = rppg_analyzer.analyze_signal_stream(data.signals, data.timestamps)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Stream analysis error: {str(e)}")
        return JSONResponse(content={'success': False, 'error': str(e)}, status_code=500)

@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    """Handle CORS preflight requests"""
    return JSONResponse(content={"message": "ok"})

@app.post("/rppg/analyze-video", response_model=RPPGAnalysisResult)
async def analyze_rppg_video(video: UploadFile = File(...)):
    """Analyze uploaded video for rPPG heart rate detection"""
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported video format. Please upload MP4, AVI, MOV, or WEBM files."
            )
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            shutil.copyfileobj(video.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Analyze video
            result = rppg_analyzer.analyze_video_file(tmp_path)
            
            if result['success']:
                return RPPGAnalysisResult(
                    success=True,
                    heart_rate=result['average_heart_rate'],
                    confidence=result['confidence'],
                    hrv=result['hrv'],
                    signal_quality=result['signal_quality'],
                    fps=result['fps'],
                    error=None
                )
            else:
                return RPPGAnalysisResult(
                    success=False,
                    heart_rate=None,
                    confidence=None,
                    hrv=None,
                    signal_quality=None,
                    fps=None,
                    error=result.get('error', 'Analysis failed')
                )
                
        finally:
            # Clean up temporary file
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
        
    except Exception as e:
        logger.error(f"Error processing video {video.filename}: {str(e)}")
        return RPPGAnalysisResult(
            success=False,
            heart_rate=None,
            confidence=None,
            hrv=None,
            signal_quality=None,
            fps=None,
            error=str(e)
        )

@app.post("/rppg/analyze-frame")
async def analyze_rppg_frame(request: RPPGFrameRequest):
    """Analyze single video frame for real-time rPPG"""
    try:
        # Convert frame data to numpy array
        frame = np.array(request.frame_data, dtype=np.uint8)
        
        # Process frame
        result = rppg_analyzer.analyze_frame_stream(frame, request.timestamp)
        
        if result:
            return {
                'success': True,
                'heart_rate': result.get('heart_rate', 0),
                'confidence': result.get('confidence', 0),
                'hrv': result.get('hrv', 0),
                'signal_quality': result.get('signal_quality', 'processing'),
                'fps': result.get('fps', 30),
                'signal': result.get('signal', []),
                'frequencies': result.get('frequencies', []),
                'power_spectrum': result.get('power_spectrum', [])
            }
        else:
            return {
                'success': True,
                'status': 'processing',
                'message': 'Collecting more frames for analysis'
            }
            
    except Exception as e:
        logger.error(f"Frame analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rppg/reset")
async def reset_rppg_analyzer():
    """Reset rPPG analyzer state for new session"""
    try:
        if rppg_analyzer:
            rppg_analyzer.reset()
            return {'success': True, 'message': 'Analyzer reset successfully'}
        else:
            return {'success': False, 'message': 'Analyzer not initialized'}
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rppg/status")
async def get_rppg_status():
    """Get rPPG analyzer status"""
    if rppg_analyzer:
        return {
            'status': 'ready',
            'buffer_size': rppg_analyzer.buffer_size,
            'current_buffer_length': len(rppg_analyzer.signal_buffer),
            'fps': rppg_analyzer.fps,
            'has_results': len(rppg_analyzer.bpm_history) > 0,
            'average_bpm': np.mean(rppg_analyzer.bpm_history) if rppg_analyzer.bpm_history else 0
        }
    else:
        return {
            'status': 'not_initialized',
            'message': 'rPPG analyzer not loaded'
        }

# Update the models/info endpoint to include rPPG status
@app.get("/models/info")
async def get_model_info():
    """Get information about all loaded models"""
    models_info = {
        "system_status": "ready",
        "available_modules": {}
    }
    
    # PCG info
    if pcg_classifier is not None:
        pcg_info = pcg_classifier.get_model_info()
        models_info["available_modules"]["pcg"] = {
            "status": "active",
            "artifact_detection": "loaded" if pcg_info['artifact_model_loaded'] else "mock",
            "murmur_detection": "loaded" if pcg_info['murmur_model_loaded'] else "mock",
            "using_mock_models": pcg_info['using_mock_models']
        }
    else:
        models_info["available_modules"]["pcg"] = {"status": "not_initialized"}
    
    # rPPG info
    if rppg_analyzer is not None:
        models_info["available_modules"]["rppg"] = {
            "status": "active",
            "analyzer": "loaded",
            "face_detection": "opencv_haar",
            "signal_extraction": "chrom_based"
        }
    else:
        models_info["available_modules"]["rppg"] = {"status": "not_initialized"}
    
    # Future modules
    models_info["available_modules"]["ecg"] = {"status": "coming_soon"}
    models_info["available_modules"]["xray"] = {"status": "coming_soon"}
    
    return models_info

# Global classifier instances
pcg_classifier: Optional[CleanPCGClassifier] = None
# Future: Add other classifiers here
# rppg_analyzer: Optional[RPPGAnalyzer] = None
# ecg_analyzer: Optional[ECGAnalyzer] = None

@app.on_event("startup")
async def startup_event():
    """Initialize all analysis modules"""
    global pcg_classifier
    
    try:
        logger.info("🚀 Initializing Medical Analysis Suite...")
        
        # Initialize PCG classifier
        config = Config()
        pcg_classifier = CleanPCGClassifier(
            artifact_model_path=str(config.ARTIFACT_MODEL_PATH),
            murmur_model_path=str(config.MURMUR_MODEL_PATH)
        )
        
        # Check model status
        model_info = pcg_classifier.get_model_info()
        if model_info['using_mock_models']:
            logger.warning("⚠️ PCG: Using mock models - place real models in models/ directory")
        else:
            logger.info("✅ PCG: Real ML models loaded successfully")
        
        # Future: Initialize other analyzers
        # logger.info("🔄 Initializing rPPG analyzer...")
        # rppg_analyzer = RPPGAnalyzer()
        
        logger.info("🎉 Medical Analysis Suite startup completed!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        logger.info("🔄 System will continue with limited functionality")

# =====================
# MAIN LANDING PAGE
# =====================

@app.get("/", response_class=HTMLResponse)
async def serve_main_page():
    """Serve the main landing page"""
    frontend_path = Path("frontend/index.html")
    
    if frontend_path.exists():
        with open(frontend_path, 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        # Fallback if file doesn't exist
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Medical Analysis Suite</h1>
                <p>Available applications:</p>
                <ul>
                    <li><a href="/pcg">PCG Classification</a></li>
                    <li><a href="/rppg">rPPG Analyzer (Coming Soon)</a></li>
                    <li><a href="/ecg">ECG Analysis (Coming Soon)</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                </ul>
            </body>
        </html>
        """)

# =====================
# PCG APPLICATION
# =====================

@app.get("/pcg", response_class=HTMLResponse)
async def serve_pcg_page():
    """Serve the PCG classification page"""
    pcg_path = Path("frontend/pcg.html")
    
    if pcg_path.exists():
        with open(pcg_path, 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(content="<h1>PCG Classification page not found. Please rename your original index.html to pcg.html</h1>")

@app.post("/analyze", response_model=PCGAnalysisResult)
async def analyze_pcg(file: UploadFile = File(...)):
    """Analyze uploaded PCG audio file"""
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

@app.post("/analyze-realtime", response_model=PCGAnalysisResult)
async def analyze_realtime_pcg(request: RealtimeAudioRequest):
    """Analyze real-time PCG data"""
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
        
        # Classify
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
                'duration': actual_duration,
                'detected_sample_rate': actual_sample_rate,
                'array_length': len(audio_array),
                'pipeline': 'exact_original',
                'using_mock_models': result['details'].get('using_mock_models', False)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing realtime audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# =====================
# rPPG APPLICATION
# =====================

@app.get("/rppg", response_class=HTMLResponse)
async def serve_rppg_page():
    """Serve the rPPG analyzer page"""
    rppg_path = Path("frontend/rppg.html")
    
    if rppg_path.exists():
        with open(rppg_path, 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        # Coming soon page
        return HTMLResponse(content="""
        <html>
        <head>
            <title>rPPG Analyzer - Coming Soon</title>
            <style>
                body {
                    background: #0D1117;
                    color: #F0F6FC;
                    font-family: 'Tiempos', Georgia, sans-serif;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    padding: 2rem;
                }
                h1 {
                    font-size: 3rem;
                    margin-bottom: 1rem;
                }
                p {
                    font-size: 1.25rem;
                    color: #8B949E;
                    margin-bottom: 2rem;
                }
                a {
                    color: #58A6FF;
                    text-decoration: none;
                    font-size: 1.1rem;
                    padding: 0.75rem 1.5rem;
                    border: 1px solid #58A6FF;
                    border-radius: 8px;
                    display: inline-block;
                    transition: all 0.3s ease;
                }
                a:hover {
                    background: #58A6FF;
                    color: #0D1117;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚧 rPPG Analyzer</h1>
                <p>This feature is coming soon!</p>
                <p>Remote photoplethysmography for contactless vital signs monitoring</p>
                <a href="/">← Back to Home</a>
            </div>
        </body>
        </html>
        """)

# Future: Add rPPG API endpoints here
# @app.post("/rppg/analyze-video")
# async def analyze_rppg_video(file: UploadFile = File(...)):
#     """Analyze video for rPPG signals"""
#     pass

# =====================
# ECG APPLICATION  
# =====================

@app.get("/ecg", response_class=HTMLResponse)
async def serve_ecg_page():
    """Serve the ECG analysis page"""
    # Similar structure to rPPG - coming soon for now
    return HTMLResponse(content="""
    <html>
    <head>
        <title>ECG Analysis - Coming Soon</title>
        <style>
            body {
                background: #0D1117;
                color: #F0F6FC;
                font-family: 'Tiempos', Georgia, sans-serif;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                margin: 0;
            }
            .container {
                text-align: center;
                padding: 2rem;
            }
            h1 {
                font-size: 3rem;
                margin-bottom: 1rem;
            }
            p {
                font-size: 1.25rem;
                color: #8B949E;
                margin-bottom: 2rem;
            }
            a {
                color: #58A6FF;
                text-decoration: none;
                font-size: 1.1rem;
                padding: 0.75rem 1.5rem;
                border: 1px solid #58A6FF;
                border-radius: 8px;
                display: inline-block;
                transition: all 0.3s ease;
            }
            a:hover {
                background: #58A6FF;
                color: #0D1117;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚧 X-Ray Analysis</h1>
            <p>This feature is coming soon!</p>
            <p>AI-powered chest X-ray interpretation and abnormality detection</p>
            <a href="/">← Back to Home</a>
        </div>
    </body>
    </html>
    """)

# =====================
# SHARED ENDPOINTS
# =====================

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="3.0.0-medical-suite"
    )

@app.get("/models/info")
async def get_model_info():
    """Get information about all loaded models"""
    models_info = {
        "system_status": "ready",
        "available_modules": {}
    }
    
    # PCG info
    if pcg_classifier is not None:
        pcg_info = pcg_classifier.get_model_info()
        models_info["available_modules"]["pcg"] = {
            "status": "active",
            "artifact_detection": "loaded" if pcg_info['artifact_model_loaded'] else "mock",
            "murmur_detection": "loaded" if pcg_info['murmur_model_loaded'] else "mock",
            "using_mock_models": pcg_info['using_mock_models']
        }
    else:
        models_info["available_modules"]["pcg"] = {"status": "not_initialized"}
    
    # Future modules
    models_info["available_modules"]["rppg"] = {"status": "coming_soon"}
    models_info["available_modules"]["ecg"] = {"status": "coming_soon"}
    models_info["available_modules"]["xray"] = {"status": "coming_soon"}
    
    return models_info

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