# src/rppg/rppg_analyzer.py
import numpy as np
import logging
from typing import Dict, List, Optional
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from collections import deque
import threading

logger = logging.getLogger(__name__)

class RPPGAnalyzer:
    """Web-compatible rPPG analyzer based on your existing implementation"""
    
    def __init__(self, buffer_size=300):
        self.buffer_size = buffer_size
        self.bpm_history = deque(maxlen=self.buffer_size)
        
        # rPPG parameters (Hz)
        self.min_hr_hz = 45.0 / 60.0
        self.max_hr_hz = 180.0 / 60.0
        self.min_rr_hz = 6.0 / 60.0
        self.max_rr_hz = 30.0 / 60.0
        
    def _is_spiky(self, signal_data):
        """Heuristic to detect spiky, motion-related noise."""
        if len(signal_data) < 30 or np.std(signal_data) == 0:
            return False
        
        std_dev = np.std(signal_data)
        
        # Relaxed outlier check for robustness
        is_large_outlier = np.max(np.abs(signal_data)) > 5 * std_dev
        
        # Check for rapid changes with a fixed threshold on normalized signal
        diff_signal = np.abs(np.diff(signal_data))
        is_rapid_change = np.max(diff_signal) > 0.5
        
        return is_large_outlier or is_rapid_change

    def _calculate_hrv(self, component, fs):
        """Calculates HRV using time-domain method (RMSSD)."""
        if not component.size or fs <= 0:
            return 0
        
        # Find R-peaks
        inverted_signal = -component
        
        # FIX: Ensure distance is at least 1 to prevent errors.
        distance = int(fs * 0.5)
        if distance < 1:
            distance = 1

        peaks, _ = signal.find_peaks(inverted_signal, distance=distance) 
        
        if len(peaks) < 2:
            return 0
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(peaks) / fs
        
        if len(rr_intervals) > 1:
            diffs_squared = np.diff(rr_intervals)**2
            return np.sqrt(np.mean(diffs_squared))  
        else:
            return 0

    def _calculate_respiration_rate(self, component, fs):
        """Calculates Respiration Rate from the signal."""
        nyquist = 0.5 * fs
        low = self.min_rr_hz / nyquist
        high = self.max_rr_hz / nyquist
        
        if low >= high or high >= 1.0 or fs <= 0:
            return 0, []
        
        b, a = signal.butter(4, [low, high], btype='bandpass')
        filtered_resp_signal = signal.filtfilt(b, a, component)
        
        windowed_signal = filtered_resp_signal * np.hamming(len(filtered_resp_signal))
        fft_raw = np.fft.rfft(windowed_signal)
        fft_mag = np.abs(fft_raw)
        
        freqs = np.fft.rfftfreq(len(windowed_signal), 1.0 / fs)
        
        valid_indices = np.where((freqs >= self.min_rr_hz) & (freqs <= self.max_rr_hz))
        
        if len(valid_indices[0]) == 0:
            return 0, windowed_signal.tolist()

        peak_index = np.argmax(fft_mag[valid_indices])
        peak_freq_hz = freqs[valid_indices][peak_index]
        
        return peak_freq_hz * 60.0, windowed_signal.tolist()

    def _process_signals(self, signals, fs):
        """Processes the signals to extract rPPG data."""
        try:
            # Color Space Transformation (CHROM)
            t = np.arange(0, signals.shape[0]) * (1.0 / fs)
            X = 3 * signals[:, 0] - 2 * signals[:, 1]
            Y = 1.5 * signals[:, 0] + signals[:, 1] - 1.5 * signals[:, 2]
            alpha = np.std(X) / np.std(Y)
            processed_signal = X - alpha * Y

            # Preprocessing (Detrend, Filter, Normalize)
            detrended = signal.detrend(processed_signal)
            nyquist = 0.5 * fs
            low = self.min_hr_hz / nyquist
            high = self.max_hr_hz / nyquist
            
            if low >= high or high >= 1.0:
                filtered = detrended
            else:
                b, a = signal.butter(4, [low, high], btype='bandpass')
                filtered = signal.filtfilt(b, a, detrended)

            normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)

            # ICA to get pulsatile component
            try:
                ica = FastICA(n_components=1, max_iter=1000, tol=0.01, random_state=0)
                ica_signal = ica.fit_transform(normalized.reshape(-1, 1)).flatten()
            except Exception:
                ica_signal = normalized

            # Calculate BPM
            windowed_component = ica_signal * np.hamming(len(ica_signal))
            fft_raw = np.fft.rfft(windowed_component)
            fft_mag = np.abs(fft_raw)
            freqs = np.fft.rfftfreq(len(windowed_component), 1.0 / fs)
            
            valid_indices = np.where((freqs >= self.min_hr_hz) & (freqs <= self.max_hr_hz))
            peak_freq_hz = 0
            if len(valid_indices[0]) > 0:
                peak_index = np.argmax(fft_mag[valid_indices])
                peak_freq_hz = freqs[valid_indices][peak_index]
            
            bpm = peak_freq_hz * 60.0
            
            # Motion robustness check
            if self._is_spiky(ica_signal):
                bpm = np.mean(self.bpm_history) if len(self.bpm_history) > 0 else 0
                text = f"Motion Detected! Using Avg HR: {bpm:.1f} BPM"
            else:
                self.bpm_history.append(bpm)
                text = f"HR: {bpm:.1f} BPM"

            # Calculate HRV and Respiration Rate
            hrv = self._calculate_hrv(ica_signal, fs)
            resp_rate, resp_signal = self._calculate_respiration_rate(ica_signal, fs)
            
            # Confidence is based on the peak-to-mean ratio of the FFT
            if np.mean(fft_mag) > 0:
                confidence = np.max(fft_mag) / np.mean(fft_mag)
            else:
                confidence = 0

            return {
                'heart_rate': round(bpm, 2),
                'hrv': round(hrv, 2),
                'resp_rate': round(resp_rate, 2),
                'confidence': round(confidence, 2),
                'signal': ica_signal.tolist(),
                'resp_signal': resp_signal
            }
            
        except Exception as e:
            logger.error(f"Signal processing error: {str(e)}")
            return {
                'heart_rate': 0,
                'hrv': 0,
                'resp_rate': 0,
                'confidence': 0,
                'signal': [],
                'resp_signal': []
            }

    def analyze_signal_stream(self, signals: list, timestamps: list) -> Dict:
        """Analyze a chunk of signal data from a live stream."""
        try:
            signal_array = np.array(signals)
            time_array = np.array(timestamps)

            if len(signal_array) < 60:
                return {'success': False, 'error': 'Not enough signal data'}

            duration = time_array[-1] - time_array[0]
            fps = len(time_array) / duration if duration > 0 else 30

            result = self._process_signals(signal_array, fps)
            
            result['success'] = True
            return result

        except Exception as e:
            logger.error(f"Stream analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}