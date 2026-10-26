PID=$(cat training_logs/downstream.pid)
echo "watching downstream comparison pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== DOWNSTREAM COMPARISON FINISHED ==="
tr '\r' '\n' < training_logs/14_downstream.log | tail -n 80
