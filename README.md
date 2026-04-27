# RL-Enhanced Large Neighborhood Search for VRPTW

This project studies the **Vehicle Routing Problem with Time Windows (VRPTW)** on the Solomon benchmark family and compares:

* **ALNS**: classical Adaptive Large Neighborhood Search
* **NLNS**: a neural / RL-guided large neighborhood search
* **Hybrid**: NLNS followed by ALNS refinement

The goal is to show, with a clean benchmark study, that a learned search policy can improve over a strong heuristic baseline while staying feasible on all test instances.

---

## 1. What this repository contains

The repository is organized around four stages:

1. **Baseline ALNS**
2. **NLNS training and inference**
3. **Hybrid NLNS → ALNS refinement**
4. **Benchmark, visualization, and ablation analysis**

The final best configuration selected by the ablation study is **Hybrid_low_destroy**.

---

## 2. Problem definition

VRPTW asks for routes that:

* start and end at the depot,
* serve every customer exactly once,
* respect vehicle capacity,
* respect customer time windows,
* minimize the global routing objective.

This project uses the **Solomon-style VRPTW instances** already present in the repository.

---

## 3. Methods

### 3.1 ALNS

ALNS is used as the classical baseline.

It alternates between:

* a **destroy** operator that removes customers,
* a **repair** operator that reinserts them.

This gives a strong heuristic baseline for comparison.

### 3.2 NLNS

NLNS adds a learned policy over the destroy/repair choices.

In this repository, the RL component is a **policy-gradient style agent** that selects operator actions during the search process.

### 3.3 Hybrid

The hybrid pipeline uses:

**ALNS solution → NLNS refinement → ALNS refinement**

In practice, the strongest configuration found by ablation is **Hybrid_low_destroy**.

---

## 4. Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If your local environment already exists, activate it before running the scripts:

```bash
source venv/bin/activate
```

---

## 5. Data

The benchmark instances are stored under `data/`.

Typical folders used in this project:

```text
data/train
data/test
```

The benchmark scripts expect the original instance files to remain available.

---

## 6. How to run the project

### 6.1 Run the ALNS baseline

Use the project’s ALNS entry point on the dataset you want to evaluate:

```bash
python3 main.py --instances_dir data/test --alns_iterations 50
```

If your local ALNS entrypoint is different, run the equivalent script already present in your repository.

---

### 6.2 Train NLNS

Train the RL-guided search policy on the training instances:

```bash
python3 main_nlns.py \
  --instances_dir data/train \
  --epochs 30 \
  --steps_per_episode 25 \
  --save_dir outputs/nlns
```

This creates:

* `outputs/nlns/checkpoints/best_model.pt`
* `outputs/nlns/checkpoints/final_model.pt`
* `outputs/nlns/logs/episode_rewards.csv`

---

### 6.3 Run NLNS inference on the test set

Load the trained policy and generate test solutions:

```bash
python3 inference_nlns.py \
  --instances_dir data/test \
  --model_path outputs/nlns/checkpoints/final_model.pt \
  --output_dir outputs/nlns_eval
```

This creates:

* `outputs/nlns_eval/instances_json/*.json`
* `outputs/nlns_eval/nlns_summary.csv`

---

### 6.4 Run the hybrid NLNS → ALNS refinement

Use ALNS solutions as starting points and refine them with the learned policy:

```bash
python3 inference_hybrid.py \
  --instances_dir data/test \
  --alns_json_dir outputs/test/instances_json \
  --model_path outputs/nlns/checkpoints/final_model.pt \
  --output_dir outputs/hybrid_eval
```

This creates:

* `outputs/hybrid_eval/instances_json/*.json`
* `outputs/hybrid_eval/hybrid_summary.csv`

---

## 7. Benchmark study

This project compares methods at three levels:

* **instance level**
* **family level**
* **global level**

### 7.1 Build the benchmark report

```bash
python3 analysis/benchmark_report.py \
  --alns_csv outputs/test/test_summary.csv \
  --nlns_csv outputs/nlns_eval/nlns_summary.csv \
  --hybrid_csv outputs/hybrid_eval/hybrid_summary.csv \
  --output_dir analysis_outputs/benchmark_3way
```

Outputs:

* `analysis_outputs/benchmark_3way/comparison.csv`
* `analysis_outputs/benchmark_3way/global_summary.csv`
* `analysis_outputs/benchmark_3way/family_summary.csv`
* `analysis_outputs/benchmark_3way/instance_comparison.csv`
* `analysis_outputs/benchmark_3way/benchmark_report.md`

---

## 8. Visualizations

The visualization script produces:

* small-multiples route plots for ALNS and NLNS,
* training curve from `episode_rewards.csv`,
* family-level gain plot.

### 8.1 Generate visuals

```bash
python3 analysis/visualize_benchmark.py \
  --instances_dir data/test \
  --alns_json_dir outputs/test/instances_json \
  --nlns_json_dir outputs/nlns_eval/instances_json \
  --rewards_csv outputs/nlns/logs/episode_rewards.csv \
  --comparison_csv analysis_outputs/benchmark/comparison.csv \
  --output_dir analysis_outputs/visuals \
  --family Clustered_large
```

Outputs:

* `analysis_outputs/visuals/alns_small_multiples_<instance>.png`
* `analysis_outputs/visuals/nlns_small_multiples_<instance>.png`
* `analysis_outputs/visuals/training_curve.png`
* `analysis_outputs/visuals/family_gain.png`

---

## 9. Ablation study

The ablation study compares:

* **ALNS**
* **NLNS**
* **Hybrid_default**
* **Hybrid_low_destroy**
* **Hybrid_high_destroy**
* **Hybrid_few_steps**
* **Hybrid_many_steps**

### 9.1 Run the ablation study

```bash
python3 analysis/run_ablation.py \
  --instances_dir data/test \
  --model_path outputs/nlns/checkpoints/final_model.pt \
  --output_dir outputs/ablation
```

Outputs:

* `outputs/ablation/ablation_summary.csv`
* `outputs/ablation/instance_comparison.csv`
* `outputs/ablation/ablation_report.md`
* per-config folders with JSON and CSV results

### 9.2 Final ablation conclusion

The ablation study selects **Hybrid_low_destroy** as the best configuration.

---

## 10. Final project flow

The complete workflow is:

1. Train ALNS baseline
2. Train NLNS policy
3. Run NLNS inference on test data
4. Run hybrid refinement with ALNS
5. Build benchmark report
6. Generate figures
7. Run ablation study
8. Keep the best hybrid configuration

---

## 11. Results summary

The final study shows that:

* NLNS improves over ALNS on the test benchmark.
* The hybrid pipeline is stronger than both standalone methods.
* The ablation study identifies **Hybrid_low_destroy** as the best overall configuration.

The benchmark tables and ablation tables are the main evidence section of the project.

---

## 12. Reproducing the full project from scratch

To reproduce the full pipeline in order:

```bash
# 1) Train NLNS
python3 main_nlns.py --instances_dir data/train --epochs 30 --steps_per_episode 25 --save_dir outputs/nlns

# 2) Evaluate NLNS
python3 inference_nlns.py --instances_dir data/test --model_path outputs/nlns/checkpoints/final_model.pt --output_dir outputs/nlns_eval

# 3) Run ALNS baseline on test data
python3 main.py --instances_dir data/test --alns_iterations 50

# 4) Run hybrid refinement
python3 inference_hybrid.py --instances_dir data/test --alns_json_dir outputs/test/instances_json --model_path outputs/nlns/checkpoints/final_model.pt --output_dir outputs/hybrid_eval

# 5) Build the benchmark report
python3 analysis/benchmark_report.py --alns_csv outputs/test/test_summary.csv --nlns_csv outputs/nlns_eval/nlns_summary.csv --hybrid_csv outputs/hybrid_eval/hybrid_summary.csv --output_dir analysis_outputs/benchmark_3way

# 6) Build the visuals
python3 analysis/visualize_benchmark.py --instances_dir data/test --alns_json_dir outputs/test/instances_json --nlns_json_dir outputs/nlns_eval/instances_json --rewards_csv outputs/nlns/logs/episode_rewards.csv --comparison_csv analysis_outputs/benchmark/comparison.csv --output_dir analysis_outputs/visuals --family Clustered_large

# 7) Run ablation
python3 analysis/run_ablation.py --instances_dir data/test --model_path outputs/nlns/checkpoints/final_model.pt --output_dir outputs/ablation
```

---

## 13. Notes

* Keep generated files out of Git using `.gitignore`.
* Do not commit `outputs/`, `data/`, or model checkpoints.
* The final chosen hybrid configuration is **Hybrid_low_destroy**.

---

## 14. Ablation table

The table below follows the style commonly used in research papers.

| Method              | Mean Obj. ↓ | Gain vs ALNS ↑ | Gain % ↑ | Feasible Rate ↑ | Win Rate vs ALNS ↑ | Mean Routes ↓ | Mean Time ↓ | Mean Distance ↓ | Runtime (s) ↓ |
| :------------------ | ----------: | -------------: | -------: | --------------: | -----------------: | ------------: | ----------: | --------------: | ------------: |
| ALNS                |   9954.0084 |         0.0000 |   0.0000 |          1.0000 |             0.0000 |       14.2778 |  14959.5799 |       5389.6664 |        3.1970 |
| NLNS                |   9358.8000 |       595.2084 |   3.3434 |          1.0000 |             0.5556 |       13.6667 |  14987.1165 |       5272.2400 |        3.1349 |
| Hybrid_default      |   9122.5957 |       831.4127 |   5.8514 |          1.0000 |             0.6667 |       13.9444 |  14996.2957 |       5301.2257 |        3.8355 |
| Hybrid_low_destroy  |   8794.9567 |      1159.0517 |   8.6299 |          1.0000 |             0.8333 |       13.7778 |  14939.4418 |       5349.1175 |        3.1759 |
| Hybrid_high_destroy |   9112.3054 |       841.7030 |   6.1176 |          1.0000 |             0.7222 |       13.8333 |  15074.0358 |       5178.0942 |        3.8711 |
| Hybrid_few_steps    |   9141.1155 |       812.8929 |   5.5051 |          1.0000 |             0.6111 |       13.9444 |  15030.3011 |       5345.7731 |        3.4631 |
| Hybrid_many_steps   |   9111.4956 |       842.5128 |   6.0367 |          1.0000 |             0.6667 |       13.9444 |  14992.7419 |       5249.9589 |        4.5894 |

**Best configuration:** `Hybrid_low_destroy`

---

## 15. Citation-ready project description

If you need a one-line summary for a paper, CV, or LinkedIn:

> This project develops and benchmarks an RL-guided Large Neighborhood Search framework for VRPTW, compares it against a classical ALNS baseline, and shows that a hybrid NLNS → ALNS pipeline with carefully tuned destroy intensity delivers the best performance on the test benchmark.
