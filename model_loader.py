from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer once
model = AutoModelForSequenceClassification.from_pretrained(
    "cybersectony/phishing-email-detection-distilbert_v2.4.1"
)
tokenizer = AutoTokenizer.from_pretrained(
    "cybersectony/phishing-email-detection-distilbert_v2.4.1"
)

def predict_email(email_text):
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = predictions[0].tolist()
    labels = {
        "legitimate_email": probs[0],
        "phishing_url": probs[1],
        "legitimate_url": probs[2],
        "phishing_url_alt": probs[3]
    }
    max_label = max(labels.items(), key=lambda x: x[1])
    return {
        "prediction": max_label[0],
        "confidence": max_label[1],
        "all_probabilities": labels
    }
