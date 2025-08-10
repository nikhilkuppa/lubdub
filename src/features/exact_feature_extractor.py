import pandas as pd
import numpy as np
import warnings
import librosa
from scipy.signal import get_window, butter, filtfilt, hilbert
from scipy.fft import fft
from scipy.stats import entropy
from typing import Dict
import logging

from ..preprocessing.metadata_processor import MetadataProcessor
from ..models.s1s2_detector import S1S2Detector

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ExactFeatureExtractor:
    """
    Uses your EXACT feature extraction functions from run-task2.ipynb
    No modifications - just direct copy of your working code
    """
    
    def __init__(self):
        self.metadata_processor = MetadataProcessor()
        self.s1s2_detector = S1S2Detector()
    
    def extract_all_features(self, audio_data: np.ndarray, original_sr: int = 1000) -> Dict:
        """
        Extract features using your EXACT pipeline:
        1. Create metadata row like run_metadata.py
        2. Run S1/S2 detection 
        3. Use your exact feature extraction functions
        """
        try:
            # Step 1: Create metadata row using your exact process
            metadata_row = self.metadata_processor.create_metadata_row(
                audio_data, original_sr, "live_recording"
            )
            
            # Step 2: Run S1/S2 detection on the processed audio
            processed_audio = np.array(metadata_row['data'])
            s1s2_results = self.s1s2_detector.detect(processed_audio)
            
            # Step 3: Add S1/S2 results to metadata (completing the row structure)
            metadata_row = self.metadata_processor.add_s1s2_detection_results(
                metadata_row, s1s2_results
            )
            
            # Step 4: Convert to pandas Series (like your original DataFrame rows)
            row = self.metadata_processor.create_dataframe_row(metadata_row)
            
            # Step 5: Use your EXACT feature extraction functions
            artifact_features = self.prepare_artifact_features(row)
            murmur_features = self.prep_murmur_features(
                processed_audio, 
                s1s2_results['samples_S1']
            )
            
            return {
                'artifact_features': artifact_features,
                'murmur_features': murmur_features,
                's1s2_results': s1s2_results,
                'metadata_row': metadata_row,
                'processed_audio': processed_audio
            }
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise
    
    def extract_artifact_features(self, row):
        """
        EXACT copy of your extract_artifact_features function from run-task2.ipynb
        """
        features = []
        
        # Get the data from the row
        s1_samples = row.get('samples_S1', [])
        s2_samples = row.get('samples_S2', [])
        s1_s2_intervals = row.get('s1_s2_intervals', [])
        s2_s1_intervals = row.get('s2_s1_intervals', [])
        audio_data = row.get('data', [])
        
        # Handle missing or empty data
        if s1_samples is None:
            s1_samples = []
        if s2_samples is None:
            s2_samples = []
        if s1_s2_intervals is None:
            s1_s2_intervals = []
        if s2_s1_intervals is None:
            s2_s1_intervals = []
        if audio_data is None:
            audio_data = []
        
        # 1. Completeness metrics (most important for artifact detection)
        has_s1 = len(s1_samples) > 0
        has_s2 = len(s2_samples) > 0
        has_s1_s2_intervals = len(s1_s2_intervals) > 0
        has_s2_s1_intervals = len(s2_s1_intervals) > 0
        
        features.extend([
            float(has_s1), 
            float(has_s2), 
            float(has_s1_s2_intervals),
            float(has_s2_s1_intervals)
        ])
        
        # 2. Detection counts and ratios
        n_s1_s2_intervals = len(s1_s2_intervals) if s1_s2_intervals else 0
        n_s2_s1_intervals = len(s2_s1_intervals) if s2_s1_intervals else 0
        
        # Use the metadata columns directly
        n_s1_meta = row.get('n_S1', 0)
        n_s2_meta = row.get('n_S2', 0)
        
        features.extend([
            n_s1_s2_intervals, n_s2_s1_intervals,
            n_s1_meta, n_s2_meta
        ])
        
        # 3. S1/S2 detection consistency
        s1_s2_ratio = n_s1_meta / (n_s2_meta + 1e-10)  # Should be close to 1 for good recordings
        s1_s2_difference = abs(n_s1_meta - n_s2_meta)
        
        features.extend([s1_s2_ratio, s1_s2_difference])
        
        # 4. Signal quality metrics from metadata
        signal_mean = row.get('mean', 0)
        signal_std = row.get('std', 0)
        signal_min = row.get('min', 0)
        signal_max = row.get('max', 0)
        duration = row.get('duration', 0)
        n_samples = row.get('n_samples', 0)
        
        # Signal range and dynamic range
        signal_range = signal_max - signal_min
        dynamic_range = signal_std / (abs(signal_mean) + 1e-10)
        
        features.extend([
            signal_mean, signal_std, signal_min, signal_max,
            signal_range, dynamic_range, duration, n_samples
        ])
        
        # 5. Interval quality metrics (if intervals exist)
        if s1_s2_intervals and s2_s1_intervals:
            s1_s2_mean = np.mean(s1_s2_intervals)
            s1_s2_std = np.std(s1_s2_intervals)
            s2_s1_mean = np.mean(s2_s1_intervals)
            s2_s1_std = np.std(s2_s1_intervals)
            
            # Total cardiac cycle duration
            total_cycle = s1_s2_mean + s2_s1_mean
            
            # Physiological plausibility (normal cardiac cycle: 40bpm to 120bpm)
            is_physiological = 1.0 if 0.5 <= total_cycle <= 1.5 else 0.0
            
            # Heart rate (cycles per minute)
            heart_rate = 60.0 / total_cycle if total_cycle > 0 else 0
            is_normal_hr = 1.0 if 60 <= heart_rate <= 100 else 0.0
            
            # Interval variability (should be low for good recordings)
            all_intervals = s1_s2_intervals + s2_s1_intervals
            interval_cv = np.std(all_intervals) / (np.mean(all_intervals) + 1e-10)  # Coefficient of variation
            
            features.extend([
                s1_s2_mean, s1_s2_std, s2_s1_mean, s2_s1_std,
                total_cycle, is_physiological, heart_rate, is_normal_hr, interval_cv
            ])
        else:
            # Fill with zeros if no intervals
            features.extend([0] * 9)
        
        # 6. Advanced signal quality from raw audio (if available)
        if len(audio_data) > 0:
            audio_array = np.array(audio_data)
            
            # SNR estimation
            signal_power = np.mean(audio_array ** 2)
            
            # Estimate noise as high-frequency content
            if len(audio_array) > 1:
                diff_signal = np.diff(audio_array)
                noise_power = np.mean(diff_signal ** 2)
                snr_estimate = 10 * np.log10(signal_power / (noise_power + 1e-10))
            else:
                snr_estimate = 0
            
            # Zero crossing rate (measure of signal complexity)
            if len(audio_array) > 1:
                zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
                zcr = zero_crossings / len(audio_array)
            else:
                zcr = 0
            
            # Signal envelope variations
            if len(audio_array) > 10:
                # Simple envelope using moving average of absolute values
                window_size = min(100, len(audio_array) // 10)
                envelope = np.convolve(np.abs(audio_array), np.ones(window_size)/window_size, mode='same')
                envelope_std = np.std(envelope)
                envelope_range = np.ptp(envelope)
            else:
                envelope_std = 0
                envelope_range = 0
            
            features.extend([snr_estimate, zcr, envelope_std, envelope_range])
        else:
            features.extend([0] * 4)
        
        return np.array(features)

    def prepare_artifact_features(self, row):
        """
        EXACT copy of your prepare_artifact_features function from run-task2.ipynb
        """
        X_artifact = self.extract_artifact_features(row)
        artifact_feature_names = [
            'has_s1', 'has_s2', 'has_s1_s2_intervals', 'has_s2_s1_intervals',
            'n_s1_s2_intervals', 'n_s2_s1_intervals',
            'n_s1_meta', 'n_s2_meta', 's1_s2_ratio', 's1_s2_difference',
            'signal_mean', 'signal_std', 'signal_min', 'signal_max',
            'signal_range', 'dynamic_range', 'duration', 'n_samples',
            's1_s2_mean', 's1_s2_std', 's2_s1_mean', 's2_s1_std',
            'total_cycle', 'is_physiological', 'heart_rate', 'is_normal_hr', 'interval_cv',
            'snr_estimate', 'zero_crossing_rate', 'envelope_std', 'envelope_range'
        ]

        # we know to drop the following features from above model training analysis
        features_to_drop = [
            'has_s1',                    # Redundant with has_s1_s2_intervals
            'has_s2',                    # Redundant with has_s1_s2_intervals
            'has_s2_s1_intervals',       # Same as has_s1_s2_intervals
            'n_s2_s1_intervals',         # Redundant with n_s1_s2_intervals
            'n_s1_meta',                 # Redundant with s1_s2_ratio
            'n_s2_meta',                 # Redundant with s1_s2_ratio
            'signal_min',               # Redundant with signal_range
            'signal_max',               # Redundant with signal_range
            'n_samples',                # Linearly dependent on duration
            's2_s1_mean',               # Redundant with s1_s2_mean
            's2_s1_std',                # Redundant with s1_s2_std
            'interval_cv',             # Derivable from s1_s2_std / mean
            'zero_crossing_rate',      # Highly correlated with snr_estimate
            'envelope_std',            # Highly correlated with envelope_range
            'envelope_range',            # Redundant with signal range and signal std independently
            's1_s2_difference',                    # Redundant with has_s1_s2_intervals
            'total_cycle',                    # Redundant with has_s1_s2_intervals
            'is_physiological',       # Same as has_s1_s2_intervals
            'heart_rate',         # Redundant with n_s1_s2_intervals
            'duration',                 # Redundant with s1_s2_ratio
            'is_normal_hr'
        ]
        df_features = pd.DataFrame(X_artifact).transpose()
        df_features.columns = artifact_feature_names
        df_features_reduced = df_features.drop(columns=features_to_drop)
        X_artifact_reduced = df_features_reduced.values
        X_artifact_reduced = np.nan_to_num(X_artifact_reduced, nan=0.0, posinf=1e6, neginf=-1e6)
        return X_artifact_reduced

    def bandpass_filter(self, audio_data, fs=1000, lowcut=20, highcut=400, order=4):
        """
        EXACT copy of your bandpass_filter function from run-task2.ipynb
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, audio_data)
        return filtered_signal

    def prep_murmur_features(self, audio_data, s1_samples, sr=1000, fft_size=256, n_mfcc=5):
        """
        EXACT copy of your prep_murmur_features function from run-task2.ipynb
        """
        audio_data = self.bandpass_filter(audio_data, fs=sr)
        segments = []
        durations = []
        mfccs_all = []

        for i in range(len(s1_samples) - 1):
            start = s1_samples[i]
            end = s1_samples[i + 1]
            
            if end > start and end <= len(audio_data):
                segment = audio_data[start:end]
                if len(segment) < 10:
                    continue

                durations.append((end - start) / sr)
                segments.append(segment)

                # Compute MFCCs
                try:
                    mfcc = librosa.feature.mfcc(y=segment.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
                    mfcc_mean = np.mean(mfcc, axis=1)  # Average over time frames
                    mfccs_all.append(mfcc_mean)
                except Exception:
                    pass  # Skip segments that are too short or noisy

        if len(segments) == 0:
            return np.zeros(15 + n_mfcc)

        # --- Frequency domain features (average spectrum) ---
        padded_segments = []
        for segment in segments:
            windowed = segment * get_window('hamming', len(segment))
            padded = np.zeros(fft_size)
            cut = min(len(windowed), fft_size)
            padded[:cut] = windowed[:cut]
            padded_segments.append(padded)

        avg_spectrum = np.mean([np.abs(fft(p))[:fft_size // 2] for p in padded_segments], axis=0)
        avg_spectrum /= np.sum(avg_spectrum) + 1e-8

        freqs = np.linspace(0, sr / 2, len(avg_spectrum))
        spectral_centroid = np.sum(freqs * avg_spectrum)
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * avg_spectrum))
        spectral_rolloff = freqs[np.searchsorted(np.cumsum(avg_spectrum), 0.85)]
        spectral_entropy = entropy(avg_spectrum)
        peak_freq = freqs[np.argmax(avg_spectrum)]

        durations = np.array(durations)
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        cv_duration = std_duration / (mean_duration + 1e-8)
        n_cycles = len(durations)

        rms_energy = np.mean([np.sqrt(np.mean(s**2)) for s in segments])
        zero_crossing_rate = np.mean([((s[:-1] * s[1:]) < 0).mean() for s in segments])

        envelopes = [np.abs(hilbert(s)) for s in segments]
        envelope_range = np.mean([np.max(env) - np.min(env) for env in envelopes])
        envelope_std = np.mean([np.std(env) for env in envelopes])

        # Top 2 bins of average spectrum
        top_bins = avg_spectrum[np.argsort(avg_spectrum)[-2:][::-1]]

        # Mean MFCCs
        mfcc_features = np.mean(mfccs_all, axis=0) if len(mfccs_all) > 0 else np.zeros(n_mfcc)

        # Final feature vector (15 + n_mfcc)
        feature_vector = np.array([
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_entropy,
            peak_freq,
            mean_duration,
            std_duration,
            cv_duration,
            n_cycles,
            rms_energy,
            zero_crossing_rate,
            envelope_range,
            envelope_std,
            *top_bins,
            *mfcc_features
        ])
        return feature_vector.reshape(1, -1)