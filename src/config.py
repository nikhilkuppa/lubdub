import os
from pathlib import Path

class Config:
    """Configuration settings for the PCG classification system"""
    
    # Model paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    
    # Try fixed/retrained models first, fallback to originals
    @classmethod
    def get_artifact_model_path(cls):
        """Get the best available artifact model"""
        possible_paths = [
            # cls.MODELS_DIR / "artifact_detector_fixed.pkl",
            # cls.MODELS_DIR / "artifact_detector_retrained.pkl", 
            cls.MODELS_DIR / "artifact_detector_randomforest.pkl"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return cls.MODELS_DIR / "artifact_detector_randomforest.pkl"  # Default
    
    @classmethod  
    def get_murmur_model_path(cls):
        """Get the best available murmur model"""
        possible_paths = [
            # cls.MODELS_DIR / "murmur_detector_fixed.joblib",
            # # cls.MODELS_DIR / "murmur_detector_retrained.joblib",
            cls.MODELS_DIR / "murmur_healthy_gradient_boosting.joblib"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        return cls.MODELS_DIR / "murmur_healthy_gradient_boosting.joblib"  # Default
    
    # Properties for easy access
    @property
    def ARTIFACT_MODEL_PATH(self):
        return self.get_artifact_model_path()
    
    @property
    def MURMUR_MODEL_PATH(self):
        return self.get_murmur_model_path()
    
    # Audio processing parameters
    TARGET_SAMPLE_RATE = 1000
    MIN_AUDIO_LENGTH = 5  # seconds
    MAX_AUDIO_LENGTH = 30  # seconds
    
    # Signal processing parameters
    BANDPASS_LOW = 25
    BANDPASS_HIGH = 400
    FILTER_ORDER = 5
    
    # Wavelet denoising parameters
    WAVELET = 'db4'
    WAVELET_LEVELS = 4
    
    # Peak detection parameters
    ENVELOPE_WINDOW = 50
    ENVELOPE_SIGMA = 20
    MIN_PEAK_DISTANCE = 200
    
    # Feature extraction parameters
    FFT_SIZE = 256
    N_MFCC = 5
    
    # Quality thresholds
    MIN_SNR = 10  # dB
    MIN_S1S2_COUNT = 3
    MAX_HEART_RATE = 180
    MIN_HEART_RATE = 40
    
    # API settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.wav', '.flac', '.mp3'}
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Environment settings
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = ENVIRONMENT == 'development'
    
    @classmethod
    def get_model_status(cls):
        """Check which models are available"""
        artifact_path = cls.get_artifact_model_path()
        murmur_path = cls.get_murmur_model_path()
        
        return {
            'artifact_model': {
                'path': str(artifact_path),
                'exists': artifact_path.exists(),
                'type': 'fixed' if 'fixed' in str(artifact_path) else 
                       'retrained' if 'retrained' in str(artifact_path) else 'original'
            },
            'murmur_model': {
                'path': str(murmur_path), 
                'exists': murmur_path.exists(),
                'type': 'fixed' if 'fixed' in str(murmur_path) else
                       'retrained' if 'retrained' in str(murmur_path) else 'original'
            }
        }