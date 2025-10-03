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

    def __init__(self, model_name='distilbert-base-uncased', n_features=19, dropout=0.5):
        super(HybridPhishingDetector, self).__init__()

        # Transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_dim = self.transformer.config.hidden_size

        # Feature processing with stronger regularization (updated architecture from mail.ipynb)
        self.feature_bn = nn.BatchNorm1d(n_features)
        self.feature_fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Combined classifier with increased dropout (updated architecture from mail.ipynb)
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
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

# Enhanced Feature Extractor Class (Copied from mail.ipynb)
class EnhancedFeatureExtractor:
    """Extract enhanced hand-crafted features that catch phishing patterns (19 features)"""
    def __init__(self):
        self.urgency_words = ['urgent', 'immediately', 'asap', 'expire', 'expiring',
                              'expires', 'deadline', 'hurry', 'rush', 'quick', 'now', 'act now']
        self.threat_words = ['suspend', 'suspended', 'block', 'blocked', 'deactivate',
                            'deactivated', 'locked', 'freeze', 'frozen', 'terminate', 'close', 'disabled']
        self.action_words = ['click', 'verify', 'confirm', 'update', 'validate',
                            'authenticate', 'secure', 'restore', 'unlock', 'download']
        self.asu_legitimate_domains = ['asu.edu', 'my.asu.edu', 'students.asu.edu', 'canvas.asu.edu',
                                       'asurite.asu.edu', 'parking.asu.edu', 'housing.asu.edu']
        self.asu_fake_domains = ['asu.com', 'asu.net', 'asu.co', 'assu.edu', 'aasu.edu',
                                'arizona-state.com', 'asu-admin.com', 'my-asu.com', 'asu-verify.tk']
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top',
                               '.click', '.pw', '.cc', '.loan', '.download', '.zip']
        self.free_email_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                                     'aol.com', 'mail.com', 'protonmail.com', 'icloud.com']
        self.typosquatting_patterns = ['micros0ft', 'g00gle', 'paypa1', 'amaz0n', 'app1e',
                                       'assu', 'aasu', 'azsu', 'azu']
        self.sensitive_terms = ['ssn', 'social security', 'password', 'pin', 'credit card',
                               'bank account', 'routing number', 'account number']

    def extract(self, text, sender=''):
        """Extract all 19 features from email text in the order required by the model"""
        if not isinstance(text, str):
            text = ''
        text_lower = text.lower()
        features = {}

        # Features 1-3: Basic pattern counts
        features['urgency_count'] = sum(word in text_lower for word in self.urgency_words)
        features['threat_count'] = sum(word in text_lower for word in self.threat_words)
        features['action_count'] = sum(word in text_lower for word in self.action_words)

        # Features 4-7: URL analysis
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        features['url_count'] = len(re.findall(url_pattern, text))
        features['has_url'] = 1 if features['url_count'] > 0 else 0

        obfuscated_pattern = r'h[x*]{2}p[s]?://'
        features['obfuscated_url'] = 1 if re.search(obfuscated_pattern, text_lower) else 0
        features['typosquatting'] = 1 if any(typo in text_lower for typo in self.typosquatting_patterns) else 0

        # Features 8-10: Text characteristics
        features['length'] = min(len(text) / 1000.0, 10.0)
        features['exclamation_marks'] = min(text.count('!'), 10)

        capitals = sum(1 for c in text if c.isupper())
        features['capital_ratio'] = capitals / len(text) if len(text) > 0 else 0

        # Features 11-12: ASU-specific domain analysis
        asu_mentioned = any(term in text_lower for term in ['asu', 'arizona state', 'sun devil'])
        if sender and asu_mentioned:
            sender_domain = sender.split('@')[-1].lower() if '@' in sender else sender.lower()
            is_legit_asu = any(domain in sender_domain for domain in self.asu_legitimate_domains)
            features['asu_domain_mismatch'] = 1 if not is_legit_asu else 0
            is_free_provider = any(provider in sender_domain for provider in self.free_email_providers)
            features['asu_from_free_email'] = 1 if (not is_legit_asu and is_free_provider) else 0
        else:
            features['asu_domain_mismatch'] = 0
            features['asu_from_free_email'] = 0

        # Features 13-16: Advanced pattern detection
        features['known_fake_domain'] = 1 if any(fake in text_lower for fake in self.asu_fake_domains) else 0
        features['suspicious_tld'] = 1 if any(tld in text_lower for tld in self.suspicious_tlds) else 0
        features['sensitive_info_request'] = 1 if any(term in text_lower for term in self.sensitive_terms) else 0

        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        features['ip_address_url'] = 1 if re.search(ip_pattern, text) else 0

        # Features 17-19: Combined indicators
        features['urgency_threat_combo'] = 1 if (features['urgency_count'] >= 2 and features['threat_count'] >= 1) else 0

        combo_score = (features['urgency_count'] * 0.3 + features['threat_count'] * 0.3 +
                      features['action_count'] * 0.2 + features['url_count'] * 0.2)
        features['phishing_combo_score'] = min(combo_score, 5.0)

        if len(text) > 0:
            total_indicators = (features['urgency_count'] + features['threat_count'] +
                              features['action_count'] + features['url_count'])
            features['indicator_density'] = (total_indicators / len(text)) * 100
        else:
            features['indicator_density'] = 0

        # Return all 19 features in the expected order
        return np.array([
            features['urgency_count'], features['threat_count'], features['action_count'],
            features['url_count'], features['has_url'], features['obfuscated_url'],
            features['typosquatting'], features['length'], features['exclamation_marks'],
            features['capital_ratio'], features['asu_domain_mismatch'], features['asu_from_free_email'],
            features['known_fake_domain'], features['suspicious_tld'], features['sensitive_info_request'],
            features['ip_address_url'], features['urgency_threat_combo'], features['phishing_combo_score'],
            features['indicator_density']
        ], dtype=np.float32)

# Load Model Weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print(f"Loading model weights from: {MODEL_PATH}")

# Note: n_features=19 and dropout=0.5 are standard values from the training script
model = HybridPhishingDetector(n_features=19, dropout=0.5)

# The fix: This should now load without the RuntimeError because the classes match.
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
feat_ext = EnhancedFeatureExtractor()

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
