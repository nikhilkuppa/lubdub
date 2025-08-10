import numpy as np
import joblib
import pickle
from typing import Dict, Optional
import logging
from pathlib import Path

from ..config import Config
from ..features.exact_feature_extractor import ExactFeatureExtractor

logger = logging.getLogger(__name__)

class CleanPCGClassifier:
    """
    Clean PCG classifier that uses your EXACT pipeline:
    1. Creates metadata exactly like run_metadata.py
    2. Uses your exact feature extraction from run-task2.ipynb
    3. Feeds features to your trained models
    """
    
    def __init__(self, artifact_model_path: str, murmur_model_path: str):
        """
        Initialize classifier with your trained models
        """
        self.artifact_model = None
        self.murmur_model = None
        self.feature_extractor = ExactFeatureExtractor()
        self.use_mock_models = False
        
        self._load_models(artifact_model_path, murmur_model_path)
    
    def _load_models(self, artifact_path: str, murmur_path: str):
        """Load your trained models"""
        try:
            # Load artifact detection model
            if Path(artifact_path).exists():
                try:
                    self.artifact_model = joblib.load(artifact_path)
                    logger.info(f"✅ Loaded artifact model from {artifact_path}")
                except:
                    with open(artifact_path, 'rb') as f:
                        self.artifact_model = pickle.load(f)
                    logger.info(f"✅ Loaded artifact model (pickle) from {artifact_path}")
            else:
                logger.warning(f"⚠️ Artifact model not found: {artifact_path}")
                self.use_mock_models = True
            
            # Load murmur detection model
            if Path(murmur_path).exists():
                try:
                    self.murmur_model = joblib.load(murmur_path)
                    logger.info(f"✅ Loaded murmur model from {murmur_path}")
                except:
                    with open(murmur_path, 'rb') as f:
                        self.murmur_model = pickle.load(f)
                    logger.info(f"✅ Loaded murmur model (pickle) from {murmur_path}")
            else:
                logger.warning(f"⚠️ Murmur model not found: {murmur_path}")
                self.use_mock_models = True
            
            if self.use_mock_models:
                logger.info("🔄 Using mock models for testing")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            self.use_mock_models = True
            logger.info("🔄 Falling back to mock models")
    
    def classify(self, audio_data: np.ndarray, original_sr: int = 1000) -> Dict:
        """
        Classify PCG signal using your EXACT pipeline
        
        Args:
            audio_data: Raw audio signal
            original_sr: Original sample rate of the audio
            
        Returns:
            Classification results
        """
        try:
            # Extract features using your EXACT pipeline
            features = self.feature_extractor.extract_all_features(audio_data, original_sr)
            
            # Step 1: Artifact detection using your exact features
            artifact_result = self._detect_artifacts(features['artifact_features'])
            
            if artifact_result['is_artifact']:
                return self._format_artifact_result(artifact_result, features)
            
            # Step 2: Murmur vs Healthy classification using your exact features
            murmur_result = self._detect_murmur(features['murmur_features'])
            
            # Step 3: Calculate heart rate and quality metrics
            s1s2_results = features['s1s2_results']
            heart_rate = self._calculate_heart_rate(s1s2_results['samples_S1'])
            signal_quality = self._assess_signal_quality(features, heart_rate)
            
            return {
                'classification': 'murmur' if murmur_result['has_murmur'] else 'healthy',
                'confidence': murmur_result['confidence'],
                'heart_rate': heart_rate,
                's1_count': len(s1s2_results['samples_S1']),
                's2_count': len(s1s2_results['samples_S2']),
                'signal_quality': signal_quality,
                'processing_time_ms': 0,  # Will be calculated by API
                'details': {
                    'artifact_score': artifact_result['confidence'],
                    'murmur_score': murmur_result['confidence'],
                    's1s2_intervals': {
                        's1_s2': s1s2_results['s1_s2_intervals'],
                        's2_s1': s1s2_results['s2_s1_intervals']
                    },
                    'using_mock_models': self.use_mock_models,
                    'metadata_created': True,
                    'features_extracted_with_exact_pipeline': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return self._format_error_result(str(e))
    
    def _detect_artifacts(self, artifact_features: np.ndarray) -> Dict:
        """Detect artifacts using your trained model"""
        try:
            if self.artifact_model is not None and not self.use_mock_models:
                # Use your trained model with your exact features
                prediction = self.artifact_model.predict(artifact_features)[0]
                
                if hasattr(self.artifact_model, 'predict_proba'):
                    probabilities = self.artifact_model.predict_proba(artifact_features)[0]
                    confidence = max(probabilities)
                else:
                    confidence = 0.8 if prediction == 1 else 0.2
                
                return {
                    'is_artifact': bool(prediction == 1),
                    'confidence': float(confidence)
                }
            else:
                # Mock logic for testing
                return {'is_artifact': False, 'confidence': 0.8}
                    
        except Exception as e:
            logger.error(f"Error in artifact detection: {str(e)}")
            return {'is_artifact': True, 'confidence': 0.9}  # Conservative
    
    def _detect_murmur(self, murmur_features: np.ndarray) -> Dict:
        """Detect murmur using your trained model"""
        try:
            if self.murmur_model is not None and not self.use_mock_models:
                # Handle your model format (with or without scaler)
                if isinstance(self.murmur_model, dict):
                    # Model with scaler (your format)
                    scaler = self.murmur_model['scaler']
                    model = self.murmur_model['model']
                    murmur_features_scaled = scaler.transform(murmur_features)
                    prediction = model.predict(murmur_features_scaled)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(murmur_features_scaled)[0]
                        confidence = max(probabilities)
                    else:
                        confidence = 0.8 if prediction == 1 else 0.2
                else:
                    # Direct model
                    prediction = self.murmur_model.predict(murmur_features)[0]
                    
                    if hasattr(self.murmur_model, 'predict_proba'):
                        probabilities = self.murmur_model.predict_proba(murmur_features)[0]
                        confidence = max(probabilities)
                    else:
                        confidence = 0.8 if prediction == 1 else 0.2
                
                return {
                    'has_murmur': bool(prediction == 1),
                    'confidence': float(confidence)
                }
            else:
                # Mock logic for testing
                return {'has_murmur': False, 'confidence': 0.8}
                    
        except Exception as e:
            logger.error(f"Error in murmur detection: {str(e)}")
            return {'has_murmur': False, 'confidence': 0.5}  # Neutral
    
    def _calculate_heart_rate(self, s1_samples: list) -> float:
        """Calculate heart rate from S1 detections"""
        if len(s1_samples) < 2:
            return 0.0
        
        # Calculate intervals between S1 sounds (cardiac cycles)
        intervals = np.diff(s1_samples) / 1000.0  # Convert to seconds (assuming 1000 Hz)
        
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
    
    def _assess_signal_quality(self, features: Dict, heart_rate: float) -> str:
        """Assess signal quality based on features and heart rate"""
        s1s2_results = features['s1s2_results']
        metadata_row = features['metadata_row']
        
        quality_score = 0
        max_score = 5
        
        # S1/S2 detection quality
        if len(s1s2_results['samples_S1']) >= Config.MIN_S1S2_COUNT:
            quality_score += 1
        if len(s1s2_results['samples_S2']) >= Config.MIN_S1S2_COUNT:
            quality_score += 1
        
        # Heart rate reasonableness
        if Config.MIN_HEART_RATE <= heart_rate <= Config.MAX_HEART_RATE:
            quality_score += 1
        
        # Signal properties
        if metadata_row['std'] > 0.01:
            quality_score += 1
        if metadata_row['duration'] >= Config.MIN_AUDIO_LENGTH:
            quality_score += 1
        
        quality_ratio = quality_score / max_score
        
        if quality_ratio >= 0.8:
            return "excellent"
        elif quality_ratio >= 0.6:
            return "good"
        elif quality_ratio >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _format_artifact_result(self, artifact_result: Dict, features: Dict) -> Dict:
        """Format result when artifact is detected"""
        return {
            'classification': 'artifact',
            'confidence': artifact_result['confidence'],
            'heart_rate': None,
            's1_count': len(features['s1s2_results']['samples_S1']),
            's2_count': len(features['s1s2_results']['samples_S2']),
            'signal_quality': 'poor',
            'processing_time_ms': 0,
            'details': {
                'artifact_score': artifact_result['confidence'],
                'reason': 'Signal contains artifacts that prevent reliable analysis',
                'using_mock_models': self.use_mock_models,
                'metadata_created': True,
                'features_extracted_with_exact_pipeline': True
            }
        }
    
    def _format_error_result(self, error_msg: str) -> Dict:
        """Format result when error occurs"""
        return {
            'classification': 'error',
            'confidence': 0.0,
            'heart_rate': None,
            's1_count': 0,
            's2_count': 0,
            'signal_quality': 'unknown',
            'processing_time_ms': 0,
            'details': {
                'error': error_msg,
                'using_mock_models': self.use_mock_models
            }
        }
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'artifact_model_loaded': self.artifact_model is not None,
            'murmur_model_loaded': self.murmur_model is not None,
            's1s2_detector': 'signal_processing_pipeline',
            'using_mock_models': self.use_mock_models,
            'feature_extraction': 'exact_original_pipeline',
            'metadata_creation': 'run_metadata_py_process',
            'status': 'ready'
        }