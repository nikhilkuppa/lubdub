
"""
S1S2 Model Wrapper
Wraps the rule-based S1S2 detector as a model for consistency
"""
import pickle
from src.models.s1s2_detector import S1S2Detector

# Create and save the S1S2 detector
detector = S1S2Detector()

# Save as a pickle file for consistency with other models
with open('models/s1s2_classifier.pkl', 'wb') as f:
    pickle.dump(detector, f)

print("S1S2 model wrapper created successfully!")
