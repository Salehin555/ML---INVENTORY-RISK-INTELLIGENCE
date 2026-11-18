cd "$(dirname "$0")"
for st in base "low lift" "high lift" "low recovery"; do
  python evaluate3.py "$st" > "outputs/log_eval_${st// /_}.txt" 2>&1 &
done
wait
for st in "high recovery" "no decline" "capped terminal"; do
  python evaluate3.py "$st" > "outputs/log_eval_${st// /_}.txt" 2>&1 &
done
python planner_select.py > outputs/log_plansel.txt 2>&1 &
wait
tail -n 2 outputs/log_eval_*.txt
echo EVALDONE
