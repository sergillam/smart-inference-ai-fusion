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
  --output-dir results/case4 \
  --log-dir logs/case4 \
  --resume
```

## Interpretation Guide
- `delta_abs_mean < 0`: degradation vs baseline
- `delta_abs_mean > 0`: improvement vs baseline
- Lower `quantization_mse_mean` indicates better numeric fidelity
- Higher `compression_ratio_mean` indicates stronger compression
- `p_value_holm < 0.05` indicates statistically significant change

## Memory Accounting Note
- `baseline_memory_bytes` and `quantized_memory_bytes` are theoretical footprint estimates based on
  tensor shape and target bit-width for the quantized representation.
- For model quantization, estimators are dequantized back to floating-point for inference stability,
  so these fields should be interpreted as compressed-storage estimates, not live Python object RSS.
