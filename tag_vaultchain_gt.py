import json
from pathlib import Path

p = Path("eval/ground_truth.json")
data = json.loads(p.read_text())

for q in data["questions"]:
    q["startup_id"] = "vaultchain"

p.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"tagged {len(data['questions'])} entries with startup_id='vaultchain'")
