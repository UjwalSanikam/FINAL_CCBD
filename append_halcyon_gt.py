import json
from pathlib import Path

gt_path = Path("eval/ground_truth.json")
new_entries_path = Path("halcyon_gt_entries.json")

gt_data = json.loads(gt_path.read_text())
new_entries = json.loads(new_entries_path.read_text())

existing_ids = {e["id"] for e in gt_data["questions"]}
overlap = existing_ids & {e["id"] for e in new_entries}
if overlap:
    raise SystemExit(f"Refusing to append — duplicate IDs already present: {overlap}")

gt_data["questions"].extend(new_entries)
gt_path.write_text(json.dumps(gt_data, indent=2), encoding="utf-8")
print(f"total entries now: {len(gt_data['questions'])}")
