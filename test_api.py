import requests

# Replace this with your actual FastAPI server URL and port
url = "http://localhost:8000/predict"

# Example suspicious email
email_text = {
    "email": "Dear user, your account has been suspended. Click here to verify your identity: http://phishing-site.com"
}

try:
    response = requests.post(url, json=email_text)
    result = response.json()

    print("✅ Server Response:")
    print(f"Prediction: {result.get('prediction')}")
    print(f"Confidence: {result.get('confidence'):.2%}")
    print("\nAll Probabilities:")
    for label, prob in result.get("all_probabilities", {}).items():
        print(f"{label}: {prob:.2%}")

except Exception as e:
    print("❌ Error:", e)
