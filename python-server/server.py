# ============================================================================
# FLASK API SERVER - Phishing Email Detection
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore') # Ignore warnings during model loading/use

app = Flask(__name__)
CORS(app)  # Allow requests from browser extension

# ============================================================================
# Model Components Setup
# ============================================================================

# Determine the device (CPU is typical for local hosting)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DEFINITIVE ARCHITECTURE (Copied from Training Script) ---
# This class definition must perfectly match the structure used when saving the .pth file.
class HybridPhishingDetector(nn.Module):
    """
    Hybrid Architecture:
    - DistilBERT for contextual understanding
    - Engineered features for explicit pattern matching
    - Combined through dense layers for final prediction
    """

    def __init__(self, model_name='distilbert-base-uncased', n_features=10, dropout=0.5):
        super(HybridPhishingDetector, self).__init__()

        # Transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_dim = self.transformer.config.hidden_size

        # Feature processing with stronger regularization
        # NOTE: Keys like 'feature_bn' and 'feature_fc' are derived from this structure.
        self.feature_bn = nn.BatchNorm1d(n_features)
        self.feature_fc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Combined classifier with increased dropout
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, features):
        # Get transformer embeddings
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = transformer_output.last_hidden_state[:, 0, :]  # [CLS] token

        # Process engineered features
        # Note: The saved model requires the feature processing layers to be called explicitly
        features_processed = self.feature_bn(features)
        features_processed = torch.relu(self.feature_fc(features_processed))

        # Combine and classify
        combined = torch.cat([pooled_output, features_processed], dim=1)
        logits = self.classifier(combined)

        return logits

# Feature Extractor Class (Copied from Training Script)
class FeatureExtractor:
    """Extract hand-crafted features that catch phishing patterns"""
    def __init__(self):
        # Ensure these lists match the training script exactly
        self.urgency_words = ['urgent', 'immediately', 'asap', 'expire', 'expiring',
                              'expires', 'deadline', 'hurry', 'rush', 'quick', 'now']
        self.threat_words = ['suspend', 'blocked', 'deactivate',
                            'deactivated', 'locked', 'freeze', 'frozen', 'terminate']
        self.action_words = ['click', 'verify', 'confirm', 'update', 'validate',
                            'authenticate', 'secure', 'restore', 'unlock']
        self.typosquatting = ['micros0ft', 'g00gle', 'paypa1', 'amaz0n', 'app1e']

    def extract(self, text):
        """Extract all 10 features from email text in the order required by the model"""
        text_lower = text.lower()
        features = {}

        # The 10 features must be extracted and returned in the precise order 
        # that the model expects them (order determined by the training script).
        
        # 1. Urgency count
        features['urgency_count'] = sum(word in text_lower for word in self.urgency_words)

        # 2. Threat count
        features['threat_count'] = sum(word in text_lower for word in self.threat_words)

        # 3. Action count
        features['action_count'] = sum(word in text_lower for word in self.action_words)

        # 4. URL count (using the training script's pattern)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        features['url_count'] = len(re.findall(url_pattern, text))

        # 5. Has URL (binary)
        features['has_url'] = 1 if features['url_count'] > 0 else 0

        # 6. Obfuscated URL (hxxp://, h**p://)
        obfuscated_pattern = r'h[x*]{2}p[s]?://'
        features['obfuscated_url'] = 1 if re.search(obfuscated_pattern, text_lower) else 0

        # 7. Typosquatting
        features['typosquatting'] = 1 if any(typo in text_lower for typo in self.typosquatting) else 0

        # 8. Length (normalized to 1000)
        features['length'] = len(text)

        # 9. Exclamation marks (normalized to 10)
        features['exclamation_marks'] = text.count('!')

        # 10. Capital ratio
        capitals = sum(1 for c in text if c.isupper())
        features['capital_ratio'] = capitals / len(text) if len(text) > 0 else 0

        # NOTE: The FeatureExtractor in the training script returns a DataFrame, 
        # but here we need a numpy array of just the values in the correct order.
        return np.array([
            features['urgency_count'], features['threat_count'], features['action_count'],
            features['url_count'], features['has_url'], features['obfuscated_url'],
            features['typosquatting'], features['length'], features['exclamation_marks'],
            features['capital_ratio']
        ], dtype=np.float32)

# Load Model Weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_phishing_detector.pth')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print(f"Loading model weights from: {MODEL_PATH}")

# Note: n_features=10 and dropout=0.5 are standard values from the training script
model = HybridPhishingDetector(n_features=10, dropout=0.5)

# The fix: This should now load without the RuntimeError because the classes match.
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
feat_ext = FeatureExtractor()

print("✓ Deep Learning Model loaded and ready!")

# ============================================================================
# API Endpoint
# ============================================================================

@app.route('/analyze-email', methods=['POST'])
def analyze_email():
    """
    Receives email data from the Chrome extension and returns a prediction.
    """
    try:
        # 1. Get email text from request
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Invalid JSON format received'}), 400
        
        # Combine subject and body into one string for the model
        email_text = f"Subject: {data.get('subject', '')} Body: {data.get('body', '')}"
        
        if not email_text:
            return jsonify({'error': 'No email text provided for analysis'}), 400
        
        # --- Prediction Pipeline ---
        
        # 2. Extract features
        features = feat_ext.extract(email_text)
        
        # Note: The training script scales features using StandardScaler. 
        # The deployment environment needs to use the same scaled values. 
        # Since we don't have the saved scaler object, we MUST skip scaling for now, 
        # which will impact accuracy. For production, the scaler must be saved and loaded here.
        
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 3. Tokenize
        inputs = tokenizer(email_text, return_tensors="pt", truncation=True, 
                          max_length=512, padding='max_length')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 4. Predict
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'], features_tensor)
            probs = torch.softmax(outputs, dim=1)[0] # Get probabilities for [Legit (0), Phish (1)]
            prediction_label = torch.argmax(outputs, dim=1).item()
        
        # 5. Format Output
        
        # Prediction label 1 is Phishing
        is_phishing = bool(prediction_label == 1)
        phish_prob = float(probs[1].item())
        legit_prob = float(probs[0].item())
        
        confidence = phish_prob if is_phishing else legit_prob
        
        if is_phishing:
            risk = 'HIGH' if phish_prob > 0.8 else 'MEDIUM'
            message = f"Phishing detected!"
        else:
            risk = 'LOW'
            message = f"Email looks safe."

        # Response format required by the Chrome extension
        result = {
            'is_phish': is_phishing,
            'confidence': confidence,
            'message': message,
            'risk_level': risk
        }
        
        print(f"\nPrediction Result: {result['message']} (Confidence: {result['confidence']:.2f})")
        
        return jsonify(result)
    
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Model file missing. {e}")
        return jsonify({'error': 'Internal server error: Model not found.'}), 500
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# ============================================================================
# Run Server
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Phishing Detection API Server - READY")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
