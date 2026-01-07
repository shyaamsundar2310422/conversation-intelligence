import os
import json
from datetime import datetime
import yaml

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# --------------------
# Confidence decay config
# --------------------
RULE_CONFIDENCE_DECAY = {
    "sarcasm_override": 0.15,
    "intent_sentiment_conflict": 0.12,
    "angry_neutral_override": 0.10,
    "complaint_neutral_override": 0.08,
    "outcome_override": 0.07,
    "neutral_phrase": 0.05,
}

DEFAULT_RULE_DECAY = 0.06
MIN_CONFIDENCE = 0.60

# --------------------
# Logging setup
# --------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rule_hits.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

def log_rule_hit(payload: dict):
    payload["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# --------------------
# App setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Conversation Intelligence API")

# --------------------
# Load models
# --------------------
emotion_tokenizer = XLMRobertaTokenizer.from_pretrained("models/emotion_xlm_roberta_final")
emotion_model = XLMRobertaForSequenceClassification.from_pretrained(
    "models/emotion_xlm_roberta_final"
).to(device).eval()

sentiment_tokenizer = XLMRobertaTokenizer.from_pretrained("models/sentiment_xlm_roberta_final")
sentiment_model = XLMRobertaForSequenceClassification.from_pretrained(
    "models/sentiment_xlm_roberta_final"
).to(device).eval()

intent_tokenizer = XLMRobertaTokenizer.from_pretrained("models/intent_xlm_roberta_final")
intent_model = XLMRobertaForSequenceClassification.from_pretrained(
    "models/intent_xlm_roberta_final"
).to(device).eval()

# --------------------
# Load rules.yaml
# --------------------
with open("rules.yaml", "r", encoding="utf-8") as f:
    RULES_CONFIG = yaml.safe_load(f)

RULES_VERSION = RULES_CONFIG.get("rules_version", "unknown")

NEUTRAL_PHRASES = RULES_CONFIG.get("neutral_phrases", [])
OUTCOME_PHRASES = RULES_CONFIG.get("outcome_phrases", [])
SARCASM_PHRASES = RULES_CONFIG.get("sarcasm_phrases", [])
DYNAMIC_RULES = RULES_CONFIG.get("rules", [])

# --------------------
# Request schema
# --------------------
class TextRequest(BaseModel):
    text: str

# --------------------
# Model helper
# --------------------
def run_model(tokenizer, model, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()

    return {
        "label": model.config.id2label[pred_id],
        "confidence": min(float(probs[pred_id]), 0.99)
    }

def predict_emotion(text):
    return run_model(emotion_tokenizer, emotion_model, text)

def predict_sentiment(text):
    return run_model(sentiment_tokenizer, sentiment_model, text)

def predict_intent(text):
    return run_model(intent_tokenizer, intent_model, text)

# --------------------
# Consistency + decision logic
# --------------------
def apply_consistency_rules(text, emotion, sentiment, intent):
    rules_triggered = []

    emotion_label = emotion["label"]
    sentiment_label = sentiment["label"]
    intent_label = intent["label"]
    text_lower = text.lower()

    # Neutral phrase
    if any(p in text_lower for p in NEUTRAL_PHRASES):
        sentiment_label = "Neutral"
        rules_triggered.append("neutral_phrase")

    # Outcome rule
    if any(p in text_lower for p in OUTCOME_PHRASES) and emotion_label in ["Sad", "Calm"]:
        sentiment_label = "Neutral"
        rules_triggered.append("outcome_override")

    # Sarcasm rule
    if sentiment_label == "Positive" and any(p in text_lower for p in SARCASM_PHRASES):
        sentiment_label = "Negative"
        rules_triggered.append("sarcasm_override")

    # YAML-driven rules
    for rule in DYNAMIC_RULES:
        cond = rule.get("if", {})
        then = rule.get("then", {})

        emotion_ok = (
            ("emotion" not in cond or cond["emotion"] == emotion_label) and
            ("emotion_in" not in cond or emotion_label in cond["emotion_in"])
        )
        sentiment_ok = ("sentiment" not in cond or cond["sentiment"] == sentiment_label)
        intent_ok = ("intent" not in cond or cond["intent"] == intent_label)

        if emotion_ok and sentiment_ok and intent_ok:
            if "sentiment" in then:
                sentiment_label = then["sentiment"]
            if "intent" in then:
                intent_label = then["intent"]
            rules_triggered.append(rule["name"])

    # --------------------
    # Confidence decay logic
    # --------------------
    base_confidence = sentiment["confidence"]
    total_decay = 0.0
    decay_breakdown = {}

    for rule_name in rules_triggered:
        decay = RULE_CONFIDENCE_DECAY.get(rule_name, DEFAULT_RULE_DECAY)
        decay_breakdown[rule_name] = decay
        total_decay += decay
    
    confidence_source = (
    "rule_adjusted" if len(rules_triggered) > 0 else "model"
)


    final_confidence = max(
        round(base_confidence - total_decay, 2),
        MIN_CONFIDENCE
    )

    priority = "High" if intent_label == "Action_Urgent" else "Normal"
    customer_state = "At Risk" if sentiment_label == "Negative" else "Stable"

    return {
        "emotion": emotion,
        "sentiment": {
    "label": sentiment_label,
    "confidence": final_confidence,
    "overridden": len(rules_triggered) > 0,
    "confidence_source": confidence_source,
    "confidence_breakdown": {
        "model_confidence": round(base_confidence, 2),
        "total_rule_decay": round(total_decay, 2),
        "rule_decay_map": decay_breakdown,
        "final_confidence": final_confidence
    }
},

        "intent": {
            "label": intent_label,
            "confidence": intent["confidence"]
        },
        "priority": priority,
        "customer_state": customer_state,
        "rules_triggered": rules_triggered,
        "rules_version": RULES_VERSION
    }

# --------------------
# API endpoint
# --------------------
@app.post("/analyze")
def analyze_text(req: TextRequest):
    emotion = predict_emotion(req.text)
    sentiment = predict_sentiment(req.text)
    intent = predict_intent(req.text)

    final_output = apply_consistency_rules(
        text=req.text,
        emotion=emotion,
        sentiment=sentiment,
        intent=intent
    )

    log_rule_hit({
    "text": req.text,

    "raw": {
        "emotion": emotion["label"],
        "sentiment": sentiment["label"],
        "intent": intent["label"]
    },

    "final": {
        "sentiment": final_output["sentiment"]["label"],
        "intent": final_output["intent"]["label"]
    },

    "confidence": {
        "source": final_output["sentiment"]["confidence_source"],
        "final": final_output["sentiment"]["confidence"],
        "total_rule_decay": final_output["sentiment"]["confidence_breakdown"]["total_rule_decay"]
    },

    "rules_triggered": final_output["rules_triggered"],
    "overridden": final_output["sentiment"]["overridden"],
    "rules_version": final_output["rules_version"]
})


    return final_output
