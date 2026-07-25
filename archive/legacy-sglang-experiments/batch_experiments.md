# 1. Run experiments (~10-30 min)
./run_batch_experiments.sh

# 2. Parse logs
python parse_timing_logs.py batch_experiment_logs/

# 3. Generate plots
python plot_latency_breakdown.py batch_experiment_logs/

# 4. View results
display batch_experiment_logs/latency_breakdown_simple.png
display batch_experiment_logs/latency_breakdown_detailed.png
