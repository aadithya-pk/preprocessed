# Toe-to-Toe Benchmark Comparison Suite (`all_compare_results/`)
## Complete Union of Evaluation Figures & Metrics Across All 8 Literature Papers

This directory contains the **comprehensive union of visual and tabular evaluation techniques** published across all 8 literature baseline papers (*IEEE Transactions on Smart Grid*, *IEEE Transactions on Power Systems*, *Nature Scientific Reports*, *ICLR*, and *AAAI*).

---

## 1. Inventory: Which Literature Paper Uses Which Figure?

| Figure File | Evaluation Technique | Literature Papers Using This Figure | What It Proves For Our Champion |
| :--- | :--- | :--- | :--- |
| **`Fig1_load_tracking_winter_peak_week.png`** | 7-Day Time Series Load Tracking (Dec 23–29, 2013) | **All 8 Papers**: EMSformer (*IEEE TPWRS*), LTE Hybrid (*IEEE TSG*), TCN-QRNN (*Nature*), TimeMixer (*ICLR*), DLinear (*AAAI*), PatchTST (*ICLR*) | Proves that the Champion hugs sharp residential evening cooking/heating peaks without phase delay or amplitude attenuation. |
| **`Fig2_regime_stress_operational_breakdown.png`** | Operational Regime Stress-Test (Peak, Base, Cold, Weekend) | LTE Hybrid (*IEEE TSG May 2025*), EMSformer (*IEEE TPWRS Nov 2025*), TCN-QRNN (*Nature 2025*) | Proves that while baseline transformer errors explode during peak hours (>150 kWh MAE), Champion maintains low error (23.39 kWh) across all operational states. |
| **`Fig3_pareto_accuracy_vs_computational_latency.png`**| Accuracy vs. Latency Pareto Frontier | TimeMixer (*ICLR 2024*), DLinear (*AAAI 2023*), PatchTST (*ICLR 2023*) | Proves that our Champion achieves the lowest MAE on the entire grid (21.64 kWh) while converging in 0.28 seconds, dominating models that take hours to train. |
| **`Fig4_residual_error_distribution_boxplot.png`** | Absolute Error Boxplots (Median, IQR, Outliers) | TCN-QRNN (*Nature Sci. Rep. 2025*), LTE Hybrid (*IEEE TSG May 2025*) | Proves our Champion has the lowest median error (16.2 kWh), the tightest interquartile range (IQR), and avoids catastrophic outlier errors. |
| **`Fig5_empirical_cumulative_distribution_ecdf.png`** | Empirical Cumulative Distribution Function (ECDF) | EMSformer (*IEEE TPWRS Nov 2025*), TCN-QRNN (*Nature 2025*) | Shows $P(\text{Error} \le X)$. Proves that over 80% of our Champion's predictions have error $< 30$ kWh, outperforming all baselines. |
| **`Fig6_parity_scatter_actual_vs_predicted.png`** | Parity Alignment Scatter Plot ($y = x$ Fit) | TCN-QRNN (*Nature 2025*), LTE Hybrid (*IEEE TSG 2025*) | Displays all 5,861 test predictions against actual load. Shows the tightest clustering around the ideal 45° diagonal ($R^2 = 0.9950$). |
| **`Fig7_ensemble_weights_and_error_correlation.png`** | Learned Mixture Weights & Residual Correlation Heatmap | Ensemble Literature (*Applied Energy*, *IEEE Trans. Power Delivery*) | Explains the physical reason for superiority: LightGBM (49.8%), MSD-TCN (32.6%), and D-PABNet (17.6%) have low error cross-correlation ($r \approx 0.41$), cancelling out individual mistakes. |

---

## 2. Accompanying Paper Tables

* **`Table_1_benchmark_leaderboard.csv`**: Master ranking table across all 12 models sorted by MAE, RMSE, WAPE, $R^2$, and Compute Time.
* **`Table_2_regime_stress_test.csv`**: Operational breakdown across Peak Hours (17:00–21:00), Overnight Base Load (02:00–06:00), Cold Snaps ($T < 20$th pct), and Weekends.
* **`Table_3_statistical_significance_dm_test.csv`**: Diebold-Mariano hypothesis test statistics and $p$-values ($p < 0.0001$).
