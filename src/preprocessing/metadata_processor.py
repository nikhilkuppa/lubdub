import pandas as pd
import soundfile as sf
import numpy as np
import librosa
from scipy.signal import resample
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MetadataProcessor:
    """
    Creates metadata using your EXACT process from run_metadata.py
    This ensures feature extraction gets the same inputs as your training
    """
    
    def __init__(self):
        self.target_sr = 1000  # From your run_metadata.py
            
        
    def create_metadata_row(self, audio_data: np.ndarray, original_sr: int, filename: str = "live_recording") -> dict:
        """
        Create a metadata row using your EXACT process from run_metadata.py
        This recreates the same data structure your models were trained on
        """
        try:
            logger.info(f"🔧 Processing audio: {len(audio_data)} samples at {original_sr} Hz")
            # Step 1: Process audio exactly like your run_metadata.py
            processed_audio = self._process_audio_like_metadata_script(audio_data, original_sr)
            logger.info(f"   After processing: {len(processed_audio)} samples at {self.target_sr} Hz")
            
            # Step 2: Extract harmonic and percussive components (like your script)
            y_harmonic, y_percussive = librosa.effects.hpss(processed_audio)
            
            # Step 3: Calculate basic statistics (exactly like your script)
            n_samples = len(processed_audio)
            duration = n_samples / self.target_sr  # Note: duration after resampling
            mean = np.mean(processed_audio)
            std = np.std(processed_audio)
            min_val = np.min(processed_audio)
            max_val = np.max(processed_audio)

            logger.info(f"   Signal stats: mean={mean:.4f}, std={std:.4f}")
            
            # Step 4: Create the metadata row with same structure as your training data
            metadata_row = {
                'filename': filename,
                'label': 'unknown',  # Will be determined by classification
                'n_samples': n_samples,
                'sampling_rate': self.target_sr,
                'duration': duration,
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'in_df': False,  # No CSV timestamps for live audio
                'n_S1': 0,       # Will be filled by S1/S2 detection
                'n_S2': 0,       # Will be filled by S1/S2 detection
                'samples_S1': [],   # Will be filled by S1/S2 detection
                'samples_S2': [],   # Will be filled by S1/S2 detection
                'data': processed_audio.tolist(),  # Exactly like your metadata
                'percussive': y_percussive.tolist(),  # Like your metadata
                'harmonic': y_harmonic.tolist()       # Like your metadata
            }
            
            logger.info(f"Created metadata row: {duration:.2f}s, {n_samples} samples")
            return metadata_row
            
        except Exception as e:
            logger.error(f"Error creating metadata row: {str(e)}")
            raise
    
    def _process_audio_like_metadata_script(self, audio_data: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Process audio using your EXACT steps from run_metadata.py
        """
        
        # Convert to mono if stereo (like your script handles)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 1000 Hz exactly like your script
        if original_sr != self.target_sr:
            # Your script uses: resample(y, int(len(y) * 1000 / sr))
            target_length = int(len(audio_data) * self.target_sr / original_sr)
            processed_audio = resample(audio_data, target_length)
        else:
            processed_audio = audio_data.copy()
        
        return processed_audio.astype(np.float64)  # Match your data type
    
    def add_s1s2_detection_results(self, metadata_row: dict, s1s2_results: dict) -> dict:
        """
        Add S1/S2 detection results to metadata row
        This completes the metadata to match your training format
        """
        
        # Update with S1/S2 detection results
        metadata_row.update({
            'n_S1': len(s1s2_results.get('samples_S1', [])),
            'n_S2': len(s1s2_results.get('samples_S2', [])),
            'samples_S1': s1s2_results.get('samples_S1', []),
            'samples_S2': s1s2_results.get('samples_S2', []),
            's1_s2_intervals': s1s2_results.get('s1_s2_intervals', []),
            's2_s1_intervals': s1s2_results.get('s2_s1_intervals', [])
        })
        
        return metadata_row
    
    def create_dataframe_row(self, metadata_row: dict) -> pd.Series:
        """
        Convert metadata row to pandas Series (like your original DataFrame)
        This ensures compatibility with your feature extraction functions
        """
        return pd.Series(metadata_row)