cd "$(dirname "$0")"
for batch in "7 11 23 42 57" "123 314 808 999 2024"; do
  for s in $batch; do
    python rl2_train.py maddpg $s 6000 both per_sku no > outputs/log_maddpg_$s.txt 2>&1 &
  done
  wait
done
grep -h "done in" outputs/log_maddpg_*.txt
echo ALLDONE
