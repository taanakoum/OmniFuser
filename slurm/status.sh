#!/bin/bash
# One-screen overview of the experiments: your queued and running jobs, the state
# of every 10-fold run and suite in experiment_results/, and logs that contain errors.
# Usage, from anywhere:  bash slurm/status.sh
# It only reads files, so it may run on the login node.
cd "$(dirname "$0")/.." || exit 1

echo "== Your jobs (ST: PD waiting, R running)"
if command -v squeue >/dev/null 2>&1; then
    squeue -u "$USER" -o "%.10i %.34j %.2t %.10M %.11l %R"
else
    echo "  (squeue is not available here)"
fi

echo
echo "== Experiments"
"${PYTHON:-python3}" - <<'EOF'
import glob
import json
import os

ROOT = "experiment_results"
SUITE_SIZES = {
    "ablations": 9, "sensitivity": 22, "missing-modality": 2, "all": 31,
    "image-encoder": 4, "image-source": 4, "deduplication": 2,
}


def run_state(path):
    folds = len(glob.glob(os.path.join(path, "fold-T*.json")))
    running = sorted(
        os.path.basename(p)[len("fold-"):-len("-resume.pt")]
        for p in glob.glob(os.path.join(path, "fold-T*-resume.pt"))
    )
    if os.path.isfile(os.path.join(path, "cross-validation-results.json")):
        return "complete", folds, running
    return ("in progress" if folds or running else "started"), folds, running


suite_dirs = {os.path.dirname(p) for p in glob.glob(os.path.join(ROOT, "*", "*", "suite.json"))}
runs = []
for results in glob.glob(os.path.join(ROOT, "**", "tracking", "run.json"), recursive=True):
    path = os.path.dirname(os.path.dirname(results))
    if not any(path.startswith(s + os.sep) for s in suite_dirs):
        runs.append(path)

if not runs and not suite_dirs:
    print("  nothing started yet")
for path in sorted(runs):
    state, folds, running = run_state(path)
    note = "  (training " + ", ".join(running) + ")" if running and state != "complete" else ""
    print("  {:<46} {:<12} {:>2}/10 folds{}".format(os.path.relpath(path, ROOT), state, folds, note))

for suite in sorted(suite_dirs):
    with open(os.path.join(suite, "suite.json"), encoding="utf-8") as file:
        manifest = json.load(file)
    records = manifest.get("runs", {})
    done = sum(1 for r in records.values() if r.get("status") == "completed")
    failed = [k for k, r in records.items() if r.get("status") == "failed"]
    total = SUITE_SIZES.get(os.path.basename(os.path.dirname(suite)), "?")
    print("  {:<46} {:<12} {}/{} variants".format(
        os.path.relpath(suite, ROOT) + "/", manifest.get("status", "?"), done, total))
    for key, record in records.items():
        if record.get("status") == "running":
            state, folds, running = run_state(os.path.join(suite, record.get("output_dir", key)))
            print("      now: {} ({}/10 folds)".format(key, folds))
    for key in failed:
        print("      FAILED: " + key)
EOF

echo
echo "== Logs with an error (time-limit stops are expected and not listed; read a log's end with: tail -n 40 FILE)"
found=0
for log in experiment_results/logs/*.out; do
    [ -f "$log" ] || continue
    last=$(grep -E "Traceback|Error:|error:|CANCELLED|oom-kill" "$log" | grep -v "DUE TO TIME LIMIT" | tail -n 1)
    if [ -n "$last" ]; then
        echo "  $log: $(echo "$last" | cut -c1-110)"
        found=1
    fi
done
if [ "$found" -eq 0 ]; then echo "  none"; fi
echo
echo "== Disk use of the results: $(du -sh experiment_results 2>/dev/null | cut -f1)"
