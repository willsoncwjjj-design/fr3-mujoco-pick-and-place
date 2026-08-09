# Warehouse Agent Evaluation

## Scope

The evaluation uses `warehouse_sorting_complex_v1`: the blue object initially
blocks the red object's target slot. Each trial must clear the slot, place both
objects at their assigned destinations, and leave the buffer empty.

- Seed: `20260809`
- Red source jitter: `±0.02 m` on X/Y
- Blue blocker jitter: `±0.01 m` on X/Y
- Success denominator: every attempted trial, including runner errors
- Placement error: final XY distance to the assigned destination, successful
  trials only

## Results

| Metric | Constraint planner | Ollama `qwen2.5:7b` |
| --- | ---: | ---: |
| Trials completed | 5 / 5 | 2 / 2 |
| Wilson 95% interval | 56.6%–100% | 34.2%–100% |
| Mean high-level actions | 3.0 | 3.0 |
| Planner requests | 15 | 9 |
| Rejected plans | 0 | 3 |
| Trials requiring plan repair | 0 / 5 | 2 / 2 |
| Mean wall time | 0.45 s | 87.89 s |
| Maximum final placement error | 8.60 mm | 8.60 mm |

The two Ollama trials verify that structured plan generation, deterministic
rejection, repair feedback, and physical execution work together. The sample is
too small to estimate production reliability or compare success rates. The
observed rejection and latency values support keeping the deterministic planner
as the operational baseline and the model behind a validation boundary.

## Reproduction

```bash
python warehouse_agent_evaluate.py --planner constraint --trials 5 --seed 20260809
python warehouse_agent_evaluate.py --planner ollama --model qwen2.5:7b --trials 2 --seed 20260809
```

Frozen trial records:

- [`warehouse_agent_constraint_trials.csv`](warehouse_agent_constraint_trials.csv)
- [`warehouse_agent_ollama_trials.csv`](warehouse_agent_ollama_trials.csv)
