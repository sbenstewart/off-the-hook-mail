import requests
import json
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:5000/analyze-email"
TARGET_SENDER = "jjohnken@asu.edu"

# ============================================================================
# UTILITY TEMPLATES FOR TESTING
# ============================================================================

BRUTE_TEMPLATES = {
    # --- PHISHING SCENARIOS (Designed to be HIGH RISK) ---
    "HIGH_RISK_SUSPENSION": {
        "subject": "URGENT: Your Account Has Been LOCKED!",
        "body": "Your account access has been suspended immediately due to unauthorized login activity. Click the link below to verify your details and restore access NOW: hxxp://micros0ft.co/login/verify"
    },
    "HIGH_RISK_INVOICE": {
        "subject": "Payment Due: Invoice #98765",
        "body": "Please find the attached invoice for $1,200. The payment is due within 24 hours. Failure to pay will result in legal action. Download the full invoice here: http://invoice-verify.net/file.exe"
    },
    "HIGH_RISK_TYPOSQUAT": {
        "subject": "Final Warning from Apple",
        "body": "Your Apple ID will be deactivated! We detected a login attempt from a new device. Click this secure link to reset your app1e account password immediately: https://apple-secure-verify.com/login"
    },
    "HIGH_RISK_CEO_SCAM": {
        "subject": "Quick task for you (Need wire transfer NOW)",
        "body": "I'm in a meeting and urgently need you to process a wire transfer to a new vendor. Do not reply to this email. Send $5,000 immediately to the account number attached. I will call you later. -CEO"
    },
    
    # --- MEDIUM RISK SCENARIOS (Ambiguous / Mild Urgency) ---
    "MEDIUM_RISK_PAYMENT": {
        "subject": "Security Check: Verify Recent Payment",
        "body": "Dear valued customer, we need to confirm a recent payment. Please log in to review the transaction before 48 hours to avoid deactivation. https://banking-update.com"
    },
    "MEDIUM_RISK_PASSWORD_RESET": {
        "subject": "Password Reset Requested for Your Account",
        "body": "A request to reset the password for your service account was initiated. If this wasn't you, please click here to cancel the request: http://reset.link/cancel?token=123"
    },
    
    # --- EDGE CASE SCENARIOS (Looks Phishy but is Real/Needs Analysis) ---
    "EDGE_CASE_LEGIT_URGENCY": {
        "subject": "Your Subscription Payment Failed",
        "body": "We regret to inform you that your last payment attempt failed. Please update your payment information in your account settings soon to avoid interruption of service. Thanks, Netflix Support."
    },
    "EDGE_CASE_OBFUSCATED": {
        "subject": "New policy document",
        "body": "Please review our updated privacy document at hxxps://company-doc.com/policy."
    },
    
    # --- LOW RISK / SAFE SCENARIOS (Designed to be LOW Risk) ---
    "LOW_RISK_MEETING": {
        "subject": "Meeting Agenda for Next Week",
        "body": "Hi team, I hope this email finds you well. I'm writing to share the proposed agenda for our Q4 planning meeting next week. No action is required on your part until the meeting itself. Thanks, Sarah."
    },
    "LOW_RISK_NEWSLETTER": {
        "subject": "Weekly Newsletter: New Features Inside!",
        "body": "Welcome to our weekly newsletter! We are excited to announce several new features and updates to our platform. Read the full post on our blog."
    },
    "LOW_RISK_INTERNAL_MEMO": {
        "subject": "System Upgrade Notification",
        "body": "This is a reminder that the IT system will undergo a scheduled upgrade tonight from 10 PM to 2 AM. Please save your work and log off before the start time. Contact IT support if you have any issues."
    }
}

# ============================================================================
# TEST EXECUTION LOGIC
# ============================================================================

def run_test():
    """Iterates through templates, calls the API, and prints results."""
    print("=" * 70)
    print(f"Starting Brute Force Heuristic Test against {API_URL}")
    print("=" * 70)
    
    final_results = []
    
    for test_name, data in BRUTE_TEMPLATES.items():
        print(f"[{test_name}] Testing...")
        
        payload = {
            "subject": data['subject'],
            "senderEmail": TARGET_SENDER,
            "body": data['body']
        }
        
        start_time = time.time()
        
        try:
            # Send POST request to the running Flask server
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            elapsed_time = time.time() - start_time
            
            final_results.append({
                "test_case": test_name,
                "subject": data['subject'],
                "is_phishing": result.get('is_phish'),
                "risk_level": result.get('risk_level', 'N/A'),
                "confidence": f"{result.get('confidence', 0.0):.4f}",
                "latency_sec": f"{elapsed_time:.2f}"
            })
            
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: Could not connect or request failed. Ensure server.py is running. Details: {e}")
            final_results.append({
                "test_case": test_name,
                "subject": data['subject'],
                "is_phishing": "ERROR",
                "risk_level": "ERROR",
                "confidence": "0.0000",
                "latency_sec": "N/A"
            })
            
    print("\n\n--- SUMMARY OF RESULTS ---")
    print(json.dumps(final_results, indent=2))
    print("\nTest Complete.")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
