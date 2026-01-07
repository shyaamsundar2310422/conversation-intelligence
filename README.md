# Conversation Intelligence API

A rule-augmented conversation intelligence system that goes beyond traditional sentiment analysis by combining **emotion detection, sentiment classification, intent recognition, and explicit reasoning rules** to produce explainable, production-ready outputs.

---

## Overview

Most sentiment analysis systems rely on a single model prediction, which often fails in real-world conversations involving sarcasm, mixed signals, or neutral phrasing around negative outcomes. This project addresses those limitations by separating **emotion**, **sentiment**, and **intent** as independent signals and reconciling them through a transparent rule engine.

The system is designed to be **deterministic, explainable, and auditable**, making it suitable for real-world applications where model decisions must be understood and trusted.

---

## Key Features

- Emotion classification using transformer-based models
- Sentiment analysis with rule-based overrides
- Intent detection independent of sentiment
- Sarcasm and contradiction handling
- Rule-driven confidence adjustment
- Explicit confidence breakdowns for every decision
- Business-level signals such as priority and customer state
- Structured, JSON-based outputs

---

## Architecture

User Text
↓
Emotion Model
↓
Sentiment Model (raw)
↓
Intent Model
↓
Rule Engine (sarcasm, conflicts, outcomes)
↓
Confidence Reconciliation
↓
Final Decision + Business Signals

---

Example

### Input

json
{
"text": "Amazing service — waited an hour for cold food."
}

### OUTPUT

{
"emotion": { "label": "Happy", "confidence": 0.79 },
"sentiment": {
"label": "Negative",
"confidence": 0.84,
"overridden": true
},
"intent": { "label": "Appreciation", "confidence": 0.96 },
"customer_state": "At Risk",
"rules_triggered": ["sarcasm_override"]
}

This demonstrates how surface emotion, true sentiment, and business risk can differ in a single message.

Design Philosophy

Models provide signals, not final truth
Rules are explicit and minimal
Emotion is preserved and not force-corrected
Sentiment overrides are transparent
Confidence values are adjusted, not replaced
This approach balances machine learning flexibility with production control and explainability.

Project Structure
.
├── app.py
├── rules.yaml
├── models/
│ ├── emotion_xlm_roberta_final/
│ ├── sentiment_xlm_roberta_final/
│ └── intent_xlm_roberta_final/
├── logs/
└── test_logs.py

Model weights are intentionally excluded to keep the repository lightweight and reproducible.

Use Cases

Customer support triage
Conversation analytics
Sarcasm detection
Risk and escalation systems
Explainable AI demonstrations

License

MIT License
