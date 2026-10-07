# Chapter 9 figure list

1. **Evaluation pipeline** — embedded directly in Section 9.1 as a LaTeX boxed flow diagram (`fig:evaluation_pipeline`). It shows multi-coil input, optional POCS, compression to 12 virtual coils, frame-wise ESPIRiT, sensitivity-weighted combination, single-image degradation, validation selection, and full-data refitting.
2. **Paired PSNR distributions** — `fig4_paired_psnr_distributions.pdf`, placed in Section 9.6. It is the principal population-level result figure and displays all 20 paired series, medians, interquartile ranges, means, and 95% bootstrap intervals.
3. **Secondary-metric intervals** — `fig5_secondary_metric_intervals.pdf`, placed in Section 9.6. It reports paired mean differences and 95% bootstrap intervals for complex NMSE and magnitude SSIM using a restrained blue-grey design. The former full-colour heatmap is not used in the main paper.
4. **Qualitative reconstruction comparison** — `fig6_qualitative_temporal_comparison.pdf`, placed in Section 9.6. Each example contains the reference, sampling mask, noisy zero-filled image, four reconstructions, and jet-coloured absolute-error maps; error cells under the reference and mask remain blank.
5. **Convergence and iteration-cap sensitivity** — `fig3_convergence_sensitivity.pdf`, placed in Section 9.7.2. It shows representative relative-iterate curves and the targeted 300-versus-600 iteration PSNR comparisons.

The parameter-boundary audit in Section 9.7.1 is reported concisely in text and does not require a separate main-paper figure.
