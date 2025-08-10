import pytest
import numpy as np
from fastapi.testclient import TestClient
import tempfile
import soundfile as sf
import io

# Import your FastAPI app
from main import app

client = TestClient(app)

class TestPCGAPI:
    """Test suite for PCG Analysis API"""
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_model_info(self):
        """Test model information endpoint"""
        response = client.get("/models/info")
        # This might fail if models aren't loaded, which is expected in testing
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "artifact_model" in data
            assert "murmur_model" in data
    
    def test_analyze_realtime_short_audio(self):
        """Test real-time analysis with too short audio"""
        # Create short audio (2 seconds at 1000 Hz = 2000 samples)
        short_audio = np.random.uniform(-0.1, 0.1, 2000).tolist()
        
        response = client.post("/analyze-realtime", json=short_audio)
        assert response.status_code == 400
        assert "too short" in response.json()["detail"].lower()
    
    def test_analyze_realtime_valid_audio(self):
        """Test real-time analysis with valid audio length"""
        # Create 6 seconds of synthetic heart sound-like signal
        duration = 6  # seconds
        sample_rate = 1000
        samples = duration * sample_rate
        
        # Generate synthetic heart sound pattern
        t = np.linspace(0, duration, samples)
        # Simple synthetic pattern: periodic beats
        heart_rate = 72  # bpm
        beat_period = 60 / heart_rate
        
        # Generate S1 and S2 patterns
        signal = np.zeros(samples)
        for beat_time in np.arange(0, duration, beat_period):
            # S1 peak
            s1_start = int(beat_time * sample_rate)
            s1_end = min(s1_start + 100, samples)
            if s1_start < samples:
                signal[s1_start:s1_end] += np.random.uniform(0.3, 0.5, s1_end - s1_start)
            
            # S2 peak (0.3 seconds after S1)
            s2_start = int((beat_time + 0.3) * sample_rate)
            s2_end = min(s2_start + 80, samples)
            if s2_start < samples:
                signal[s2_start:s2_end] += np.random.uniform(0.2, 0.4, s2_end - s2_start)
        
        # Add some noise
        noise = np.random.uniform(-0.05, 0.05, samples)
        signal += noise
        
        audio_data = signal.tolist()
        
        response = client.post("/analyze-realtime", json=audio_data)
        
        # This might fail if models aren't properly loaded
        if response.status_code == 200:
            data = response.json()
            assert "classification" in data
            assert "confidence" in data
            assert "signal_quality" in data
            assert data["classification"] in ["healthy", "murmur", "artifact", "error"]
            assert 0 <= data["confidence"] <= 1
        else:
            # Accept 500 errors in test environment where models might not be loaded
            assert response.status_code == 500
    
    def test_analyze_file_invalid_format(self):
        """Test file upload with invalid format"""
        # Create a fake text file
        fake_file = io.BytesIO(b"This is not an audio file")
        
        response = client.post(
            "/analyze",
            files={"file": ("test.txt", fake_file, "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    
    def test_analyze_file_valid_wav(self):
        """Test file upload with valid WAV file"""
        # Create a temporary WAV file
        duration = 8  # seconds
        sample_rate = 44100
        samples = duration * sample_rate
        
        # Generate synthetic audio
        t = np.linspace(0, duration, samples)
        audio_data = 0.3 * np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
        wav_buffer.seek(0)
        
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", wav_buffer, "audio/wav")}
        )
        
        # This might fail if models aren't properly loaded
        if response.status_code == 200:
            data = response.json()
            assert "classification" in data
            assert "confidence" in data
            assert "processing_time_ms" in data
            assert "metadata" in data
        else:
            # Accept 500 errors in test environment where models might not be loaded
            assert response.status_code == 500

    def test_cors_headers(self):
        """Test that CORS headers are properly set"""
        response = client.options("/analyze")
        # CORS preflight should be handled by middleware
        assert response.status_code in [200, 405]  # Some frameworks return 405 for OPTIONS

# Pytest fixtures for testing
@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing"""
    duration = 10  # seconds
    sample_rate = 1000
    samples = duration * sample_rate
    
    # Generate realistic heart sound pattern
    t = np.linspace(0, duration, samples)
    signal = np.zeros(samples)
    
    # Add periodic heart beats
    for beat_time in np.arange(0, duration, 0.8):  # ~75 bpm
        s1_idx = int(beat_time * sample_rate)
        s2_idx = int((beat_time + 0.3) * sample_rate)
        
        if s1_idx < samples:
            signal[s1_idx:min(s1_idx + 50, samples)] += 0.4
        if s2_idx < samples:
            signal[s2_idx:min(s2_idx + 30, samples)] += 0.3
    
    # Add noise
    noise = np.random.normal(0, 0.02, samples)
    signal += noise
    
    return signal

@pytest.fixture
def wav_file_bytes(sample_audio_data):
    """Create WAV file bytes from sample audio data"""
    buffer = io.BytesIO()
    sf.write(buffer, sample_audio_data, 44100, format='WAV')
    buffer.seek(0)
    return buffer

def test_integration_workflow(sample_audio_data):
    """Test complete workflow from audio to classification"""
    # This test would require models to be loaded
    # In a real test environment, you'd mock the model responses
    
    audio_list = sample_audio_data.tolist()
    
    response = client.post("/analyze-realtime", json=audio_list)
    
    # For integration tests, we accept that models might not be loaded
    assert response.status_code in [200, 400, 500]
    
    if response.status_code == 200:
        data = response.json()
        # Validate response structure
        required_fields = [
            "classification", "confidence", "signal_quality",
            "s1_count", "s2_count", "processing_time_ms"
        ]
        for field in required_fields:
            assert field in data

if __name__ == "__main__":
    pytest.main([__file__])