import numpy as np
import scipy.signal
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple
import logging

from ..config import Config

logger = logging.getLogger(__name__)

class S1S2Detector:
    """Detect S1 and S2 heart sounds in PCG signals"""
    
    def __init__(self):
        self.sample_rate = Config.TARGET_SAMPLE_RATE
        self.envelope_window = Config.ENVELOPE_WINDOW
        self.envelope_sigma = Config.ENVELOPE_SIGMA
        self.min_peak_distance = Config.MIN_PEAK_DISTANCE
        
    def detect(self, audio_data: np.ndarray) -> Dict:
        """
        Detect S1 and S2 heart sounds in preprocessed audio
        
        Args:
            audio_data: Preprocessed PCG signal
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Find peaks in envelope
            envelope, peaks = self._find_envelope_peaks(audio_data)
            
            # Classify peaks as S1 or S2
            labeled_peaks = self._classify_peaks(peaks)
            
            # Extract S1 and S2 samples
            samples_S1 = [p for p, label in labeled_peaks if label == 'S1']
            samples_S2 = [p for p, label in labeled_peaks if label == 'S2']
            
            # Sort samples
            samples_S1.sort()
            samples_S2.sort()
            
            # Compute intervals
            intervals = self._compute_intervals(samples_S1, samples_S2)
            
            return {
                'envelope': envelope,
                'peaks': peaks,
                'labeled_peaks': labeled_peaks,
                'samples_S1': samples_S1,
                'samples_S2': samples_S2,
                's1_s2_intervals': intervals['s1_s2'],
                's2_s1_intervals': intervals['s2_s1']
            }
            
        except Exception as e:
            logger.error(f"Error in S1S2 detection: {str(e)}")
            # Return empty results on error
            return {
                'envelope': np.array([]),
                'peaks': np.array([]),
                'labeled_peaks': [],
                'samples_S1': [],
                'samples_S2': [],
                's1_s2_intervals': [],
                's2_s1_intervals': []
            }
    
    def _find_envelope_peaks(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks in the signal envelope"""
        # Compute envelope using Hilbert transform
        envelope = np.abs(hilbert(audio_data))
        
        # Smooth the envelope
        envelope = gaussian_filter1d(envelope, sigma=self.envelope_sigma)
        
        # Additional smoothing with moving average
        kernel = np.ones(self.envelope_window) / self.envelope_window
        smoothed_envelope = np.convolve(envelope, kernel, mode='same')
        
        # Find peaks
        peaks, _ = scipy.signal.find_peaks(
            smoothed_envelope, 
            distance=self.min_peak_distance
        )
        
        # Filter peaks based on amplitude threshold
        if len(peaks) > 0:
            threshold = np.median(smoothed_envelope) + 0.1 * np.std(smoothed_envelope)
            peaks = peaks[smoothed_envelope[peaks] > threshold]
        
        return smoothed_envelope, peaks
    
    def _classify_peaks(self, peaks: np.ndarray) -> List[Tuple[int, str]]:
        """Classify peaks as S1 or S2 based on interval patterns"""
        if len(peaks) < 2:
            return [(p, 'S1') for p in peaks]  # Default to S1 if too few peaks
        
        intervals = np.diff(peaks)
        labels = []
        
        # Initial classification based on first interval
        if len(intervals) > 0:
            if len(intervals) == 1:
                # Only two peaks - assume S1, S2
                labels = ['S1', 'S2']
            else:
                # Use interval pattern to determine starting label
                if intervals[0] < intervals[1]:
                    labels = ['S1', 'S2']
                else:
                    labels = ['S2', 'S1']
                
                # Classify remaining peaks based on interval pattern
                for i in range(1, len(intervals)):
                    last_interval = intervals[i - 1]
                    current_interval = intervals[i]
                    
                    # S1-S2 intervals are typically shorter than S2-S1
                    if current_interval > last_interval:
                        next_label = 'S1'
                    else:
                        next_label = 'S2'
                    labels.append(next_label)
                
                # Ensure we have the right number of labels
                if len(labels) < len(peaks):
                    labels.append('S1' if labels[-1] == 'S2' else 'S2')
        
        # Create labeled peaks list
        labeled_peaks = list(zip(peaks, labels[:len(peaks)]))
        return labeled_peaks
    
    def _compute_intervals(self, samples_S1: List[int], samples_S2: List[int]) -> Dict[str, List[float]]:
        """Compute intervals between S1 and S2 sounds"""
        if not samples_S1 or not samples_S2:
            return {'s1_s2': [], 's2_s1': []}
        
        # Combine and sort all samples
        all_samples = sorted(samples_S1 + samples_S2)
        
        s1_s2_intervals = []
        s2_s1_intervals = []
        
        # Calculate intervals between consecutive peaks
        for i in range(len(all_samples) - 1):
            current_sample = all_samples[i]
            next_sample = all_samples[i + 1]
            interval_duration = (next_sample - current_sample) / self.sample_rate
            
            # Determine interval type
            if current_sample in samples_S1 and next_sample in samples_S2:
                s1_s2_intervals.append(interval_duration)
            elif current_sample in samples_S2 and next_sample in samples_S1:
                s2_s1_intervals.append(interval_duration)
        
        return {
            's1_s2': s1_s2_intervals,
            's2_s1': s2_s1_intervals
        }
    
    def get_heart_rate(self, samples_S1: List[int]) -> float:
        """Calculate heart rate from S1 detections"""
        if len(samples_S1) < 2:
            return 0.0
        
        # Calculate intervals between S1 sounds (cardiac cycles)
        intervals = np.diff(samples_S1) / self.sample_rate  # Convert to seconds
        
        if len(intervals) == 0:
            return 0.0
        
        # Average cycle duration
        avg_cycle_duration = np.mean(intervals)
        
        # Convert to beats per minute
        heart_rate = 60.0 / avg_cycle_duration if avg_cycle_duration > 0 else 0.0
        
        # Sanity check for physiologically reasonable heart rate
        if heart_rate < Config.MIN_HEART_RATE or heart_rate > Config.MAX_HEART_RATE:
            logger.warning(f"Calculated heart rate {heart_rate} bpm is outside normal range")
        
        return heart_rate