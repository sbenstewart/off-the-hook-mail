from flask import Flask, request, jsonify
import time
import json
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS to allow the Chrome extension to make requests to this server
CORS(app)

@app.route('/analyze-email', methods=['POST'])
def analyze_email():
    """
    Receives email data, simulates a delay, and sends a mock response.
    """
    if request.method == 'POST':
        # Get the JSON data from the request
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        print("\nReceived Email Data:")
        print(json.dumps(data, indent=2))
        
        # --- Simulating a processing delay ---
        print("\nSimulating a 5-second processing delay...")
        time.sleep(5)  # Simulate a long-running process like an ML model
        
        # --- Analyze the data (mock logic) ---
        # A simple check for a suspicious keyword
        is_phish = "urgent" in data.get("subject", "").lower() or "verify" in data.get("body", "").lower()
        
        # Prepare the response
        if is_phish:
            response = {
                "is_phish": True,
                "confidence": 0.95,
                "message": "Potential phishing email detected due to suspicious keywords."
            }
        else:
            response = {
                "is_phish": False,
                "confidence": 0.10,
                "message": "No phishing indicators found."
            }
        
        print("\nSending response...")
        return jsonify(response)

if __name__ == '__main__':
    # The server will be accessible at http://127.0.0.1:5000
    app.run(debug=True)