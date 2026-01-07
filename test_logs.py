import json

total = 0
overridden = 0

with open("logs/rule_hits.jsonl") as f:
    for line in f:
        total += 1
        data = json.loads(line)
        if data["overridden"]:
            overridden += 1

print("Override rate:", overridden / total if total > 0 else 0)

from collections import Counter
import json

rule_counts = Counter()

with open("logs/rule_hits.jsonl") as f:
    for line in f:
        data = json.loads(line)
        rule_counts.update(data["rules_triggered"])

print(rule_counts)
        
with open("logs/rule_hits.jsonl") as f:
    for line in f:
        data = json.loads(line)
        if "sarcasm_override" in data["rules_triggered"]:
            print(data["text"])
