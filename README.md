# Jangarun-et-al-2026-Autoencoder

Autoencoder-based baselines determination of geochemical features in topsoil and subsoil. Exported from the project Notion workspace.

## Method

- **Data**: REE concentrations (La, Ce, Pr, Nd, Sm, Eu, Gd, Dy, Tb, Ho, Er, Tm, Yb, Lu) for two soil levels (Topsoil, Subsoil).
- **Preprocessing**: 80/20 train/test split, then within-split percentile-rank normalisation (no data leakage).
- **Model**: `ImprovedAutoencoder` — fully connected encoder 14→128→64→32 and symmetric decoder, with BatchNorm + LeakyReLU + Dropout(0.2) and a sigmoid output.
- **Loss**: `HybridLoss` = 0.7·MSE + 0.3·L1.
- **Optimisation**: AdamW (lr=1e-3, wd=1e-2), `ReduceLROnPlateau`, early stopping (patience=20), max 200 epochs, batch size 32.
- **Validation**: 10-fold cross-validation on the training split per soil type.
- **Thresholds**: 1000-iteration bootstrap of reconstructed values → Q1/Q3/IQR → percentile-rank thresholds converted back to PPM via the empirical CDF; outlier fences at Q1/Q3 ± 1.5·IQR (mild) and ± 3·IQR (extreme).
- **Outputs**: per-element static (matplotlib PNG) and interactive (Plotly HTML) plots showing mild/extreme outliers vs normal range, plus a CSV of thresholds and stats per soil type.

## Repository contents

- `script.py` — full preprocessing, training, threshold, and plotting pipeline.
- `RESULTS.md` — captured training logs, CV metrics, and PPM thresholds for Topsoil and Subsoil.
- `Dataset.csv` *(to be added manually)* — 775 rows of REE concentrations with `Data` and `Level` columns.
- `best_model.pt` *(to be added manually)* — final saved model weights.
- `Topsoil_<element>_thresholds.png` / `.html`, `Subsoil_<element>_thresholds.png` / `.html` *(to be added manually)* — per-element threshold plots.
- `Topsoil_thresholds_and_stats.csv`, `Subsoil_thresholds_and_stats.csv` *(to be added manually)* — exported thresholds and statistics.
- `Untitled.ipynb` *(to be added manually)* — working notebook.
