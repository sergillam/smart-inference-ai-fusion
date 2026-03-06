# SIP-Q Results Report

## Scope
This report consolidates results from Case Study 4 (SIP-Q), covering:
- supervised models (accuracy/F1)
- unsupervised models (silhouette/ARI/NMI)
- efficiency (compression ratio, memory reduction, runtime overhead)

## Reproducibility Manifest
Each result record includes:
- `seed`
- `metadata.execution_id`
- `metadata.config_hash`
- `bit_width`, `dtype_profile`, `quantization_method`

Recommended run command:

```bash
python scripts/case4.py \
  --datasets Wine Digits MakeBlobs MakeMoons \
  --algorithms KNN DT MLP MBK GMM AC \
  --bits 8 16 32 \
  --seeds 42 123 456 789 1024 \
  --output results/case4 \
  --resume
```

## Analysis Workflow
1. Generate/collect case4 JSON outputs in `results/case4`.
2. Run:

```bash
python scripts/analyze_case4_results.py \
  --results-dir results/case4 \
  --output-dir results/case4/analysis
```

3. Inspect generated artifacts:
- `summary.csv`
- `significance.csv`
- `accuracy_vs_bitwidth.png`
- `pareto_frontier.png`
- `heatmap_degradation.png`
- `bars_by_paradigm.png`

## Statistical Method
- Paired deltas computed per seed (`quantized - baseline`)
- 95% bootstrap confidence interval for mean delta
- Paired significance test per group:
  - Wilcoxon when sample is large enough and non-constant
  - one-sample t-test fallback
- Holm-Bonferroni p-value correction across comparisons

## Interpretation Guide
- `delta_abs_mean < 0`: degradation vs baseline
- `delta_abs_mean > 0`: improvement vs baseline
- Lower `quantization_mse_mean` indicates better numeric fidelity
- Higher `compression_ratio_mean` indicates stronger compression
- `p_value_holm < 0.05` indicates statistically significant change
