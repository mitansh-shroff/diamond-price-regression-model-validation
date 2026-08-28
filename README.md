# Diamond Price Prediction: Regression Modeling & Validation

A statistical modeling project on the classic 53,940-row diamonds dataset, focused less on "getting a good R²" and more on **validating that a model is actually trustworthy** — testing whether a clean multicollinearity fix is truly safe, and whether a model that looks good in aggregate is still good where it matters most.

## Overview

This project builds a regression model to predict diamond `price` from its physical and quality attributes, then rigorously validates it. The core finding: **two textbook fixes for multicollinearity (dropping features, PCA) looked clean on paper (VIF ≈ 1) but catastrophically mispriced the rarest, most valuable diamonds once tested on held-out data in real dollar terms.** Ridge regression — which keeps all features but regularizes them — matched the best raw model's accuracy while measurably stabilizing its coefficients, and was selected as the final model.

Along the way, examining the model's most influential data points (via Cook's Distance) also surfaced a data-entry error that simple range-based cleaning had missed.

**Techniques used:** EDA, ordinal encoding, log-transformation, OLS regression, VIF / multicollinearity diagnostics, PCA, Ridge regression (RidgeCV), bootstrap coefficient stability analysis, Cook's Distance influence diagnostics, train/test evaluation on back-transformed (dollar-scale) predictions.

## Dataset

`diamonds.csv` — 53,940 diamonds with 10 attributes: `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `price`, and physical dimensions `x`, `y`, `z`. This is the well-known diamonds dataset originally distributed with the `ggplot2` R package, commonly mirrored on Kaggle and other data science resources.

## Repository Structure

```
├── diamond_price_prediction.ipynb   # Full analysis: EDA → modeling → validation → insights
├── diamonds.csv                     # Raw dataset (53,940 rows)
├── requirements.txt                 # Python dependencies
└── README.md
```

The notebook and dataset are kept in the same folder so `pd.read_csv('diamonds.csv')` works immediately after cloning — no path edits needed.

## Methodology

**Part 1 — Data Cleaning & Exploration:** Removed 169 problematic rows (146 duplicates, 20 zero-dimension entries, 3 extreme outliers) out of 53,940. Encoded `cut`, `color`, and `clarity` as ordinal variables reflecting their true quality ranking. Log-transformed `price` and `carat` to address right-skew. EDA showed diamond size utterly dominates price, with cut/color/clarity's real effect masked by size until controlled for.

**Part 2 — Model Development & Validation:** Diagnosed severe multicollinearity (VIF in the hundreds) among `carat`, `x`, `y`, `z`. Built and compared five models varying the predictor set and target transform:

| Model | Predictors | Target | Approach |
|---|---|---|---|
| 1 | All 9 raw features | `price` | Baseline |
| 2 | All 9 raw features | `log(price)` | Tests log-transform |
| 3 | `carat` only (drop x/y/z) | `log(price)` | Fixes multicollinearity by dropping features |
| 4 | PCA size component | `log(price)` | Fixes multicollinearity via dimensionality reduction |
| **5** | **All 9 features, Ridge** | `log(price)` | **Fixes multicollinearity via regularization** |

All models were evaluated on a held-out 20% test set, with log-scale predictions back-transformed to dollars for a fair, apples-to-apples comparison.

**Part 3 — Interpretation:** Discusses practical implications (e.g., routing large/rare diamonds to expert appraisal rather than automated pricing), honestly documented limitations (only linear-family models tested, single train/test split, ordinal-encoding assumptions, missing potentially relevant attributes like certification and fluorescence), and concrete recommendations for extending the analysis.

## Results

| Model | Test RMSE ($) | Test MAE ($) | Test R² | Multicollinearity |
|---|---|---|---|---|
| 1: raw → price | 1,215.80 | 797.91 | 0.905 | Severe (unaddressed) |
| 2: raw → log(price) | 876.76 | 457.79 | 0.951 | Severe (unaddressed) |
| 3: carat only → log(price) | 123,179.05 | 3,248.52 | −975.8 | Fixed on paper — catastrophic in practice |
| 4: PCA size → log(price) | 6,908.50 | 1,001.76 | −2.07 | Fixed on paper — still fails badly |
| **5: Ridge (full, scaled) → log(price)** | **879.59** | **457.76** | **0.950** | **Meaningfully reduced, no accuracy cost** |

**Selected model: Model 5 (Ridge regression, full feature set, `log(price)` target).** It matches the best OLS model's accuracy while directly addressing multicollinearity — bootstrap resampling showed `z`'s coefficient instability dropping from ~42% to ~34% CV — without discarding the size information Models 3 and 4 needed but threw away.

## Key Findings

- **Diamond size (`carat`, `x`, `y`, `z`) is overwhelmingly the primary price driver**; cut/color/clarity play smaller, secondary roles.
- **A clean VIF score doesn't guarantee a multicollinearity "fix" is safe.** Models 3 and 4 both eliminated multicollinearity on paper (VIF ≈ 1) yet extrapolated wildly — off by millions of dollars — on the rarest, largest diamonds once tested properly on held-out data in dollar terms.
- **Model diagnostics can catch what data cleaning misses.** Cook's Distance flagged three training rows where a diamond's `carat` value had apparently been mistakenly copied into its `z` (depth) field — a plausible-looking value in isolation that simple range checks couldn't detect.

## How to Run

```bash
git clone https://github.com/mitansh-shroff/diamond-price-regression-model-validation
cd diamond-price-regression-model-validation
pip install -r requirements.txt
jupyter notebook diamond_price_prediction.ipynb
```

The dataset (`diamonds.csv`) is included in the repo, so the notebook runs immediately with no additional downloads.

## Limitations

- Only linear-family models (OLS, Ridge) were tested; a non-linear approach (polynomial terms, Random Forest, Gradient Boosting) might better capture the curvature seen at the high-carat extreme.
- Evaluation used a single 80/20 train/test split; exact dollar figures could shift somewhat with a different split.
- Ordinal encoding assumes equhttps://github.com/mitansh-shroff/diamond-price-regression-model-validational spacing between quality grades (e.g. Fair→Good ≈ Premium→Ideal in effect), an assumption not directly tested against one-hot encoding.
- The dataset itself is missing potentially relevant attributes (certification, fluorescence, market/region, timing) that could further improve or explain diamond pricing.

See the notebook's Part 3 for the full discussion, including practical recommendations and what additional data would strengthen the analysis.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
