# The shared experiment protocol, identical to TRAINING_RUNBOOK.md §9.
# Load it in your login terminal before submitting jobs:
#   source slurm/protocol.sh
# Commands then insert it with "${PROTOCOL[@]}" (quotes, braces and [@] included).
PROTOCOL=(
  --data-dir Nonastreda_Data --device cuda
  --epochs 30 --batch-size 32 --learning-rate 1e-4
  --input-len 96 --horizon 96 --stride 24
  --pool 1000 --image-lag 0 --prediction-head direct
  --image-encoder efficientnet_b0 --seed 42 --num-workers 4
)
