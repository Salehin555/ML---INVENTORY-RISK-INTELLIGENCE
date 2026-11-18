# Decision Value of Side Information in Inventory Obsolescence: Multistage Planning versus Multi-Agent Reinforcement Learning

Code, aggregate results, trained policies and figures for the revised manuscript.

> **Note on earlier versions of this repository.** Results previously shown here came from the first submitted
> version of the paper and are **superseded**. The following claims are **withdrawn** and are not supported by the
> revised analysis:
>
> | Earlier claim | Status in the revised paper |
> |---|---|
> | A learned transition (surrogate) model as the training and evaluation environment | Replaced by a simulator that replays each SKU's recorded stock ledger in money terms |
> | Positive episode rewards and superiority of QMIX over the baselines | Withdrawn: no learned policy improves on the firm's own decisions |
> | 88.7% lower failure rate; time to obsolescence extended from 1.15 to 6.33 months; 15-22% of working capital released | Withdrawn |
> | Significant differences between algorithms based on three seeds (p = 0.00037) | Withdrawn: ten seeds per algorithm; the algorithms cannot be separated in mean |
> | Time-to-inactivity forecast with R² of 0.87-0.93 | Withdrawn: the high R² came from items that never sell; the forecast is no longer used |
> | Structural covariates carry more than 60% of the attribution | Corrected to 41% |
>
> The material of the submitted version is kept, labelled as superseded, in `legacy_submitted_version/`.

## What the study does

A manufacturer's ERP records (81,574 SKU-months, 6,807 SKUs, 2024; labels for 2025) are used to ask whether side
information improves monthly obsolescence decisions: whether to monitor an item, stop buying it or clear its stock.

1. **Attribution versus decision value (RQ1).** SHAP attribution of a risk classifier and of a model of whether an item
   will sell again is compared with the decision value of the same covariates in clear-or-keep decisions. The outcome
   of keeping is observed; the cost of clearing follows from an assumed recovery rate.
2. **Sequential control (RQ2).** Monitoring, purchasing and clearance are linked decisions with irreversible exit.
   Controllers are compared by replaying each held-out SKU's recorded stock ledger: the firm's own decisions, a
   rule-based ladder, newsvendor and (s,S) policies (also tuned on validation SKUs), a multistage predict-then-optimize
   planner, QMIX, MAPPO and MADDPG (ten seeds each), and the exact hindsight optimum.
3. **Stability (RQ3).** Seeds, markdown, recovery and end-of-horizon valuation assumptions, influential items, and an
   out-of-year test on the 2025 labels.

## Main results (368 test SKUs, replay of the 2024 ledger)

| | Gain over the firm's decisions, million BDT [95% CI] |
|---|---|
| Hindsight optimum (bound) | +47.7 [31.7, 67.5] (18.5% of the value of the opening stock) |
| B1 rule-based ladder | -9.6 [-30.1, 2.4] |
| B5 multistage planner | -19.5 [-60.6, 13.1] |
| Newsvendor / (s,S), tuned on validation SKUs | -25.9 [-52.5, -5.3] / -29.6 [-58.5, -6.6] |
| QMIX / MAPPO / MADDPG (mean of ten seeds) | -45.8 [-75.7, -21.3] / -61.5 [-97.7, -29.9] / -56.5 [-90.7, -28.1] |

- No feasible controller improves significantly on the firm's decisions under any assumption tested.
- Liquidating dead stock adds profit; replacing the firm's orders and removing a few items that later sell cost more.
- A handful of items decides the comparison between the firm, the planner and the classical rules: without the five
  items on which it differs most from the firm, the planner earns +16.4 million [5.9, 28.5]. The learned policies
  lose broadly.
- When leftover stock is valued conservatively, most losses are no longer significant.
- Product-range covariates carry 41% of the classifier's attribution but 16-24% in a model of later sales; their
  decision value is small and not robust to the decision rule or the outcome horizon.
- QMIX varies less across seeds than MAPPO (Levene p < 0.001); against MADDPG the evidence is suggestive only.

"Profit" is the replay value: revenue minus purchases, carrying and review cost, plus liquidation revenue, over
April-November, plus the value of the stock left in November. The opening stock (BDT 258.4 million at cost) is the
same for every controller and is not charged, so the replay value is not accounting profit. All results are
conditional on the replay of the recorded demand and on the implementations tested; they do not show that the firm
would beat these controllers in deployment, or that multistage planning is inherently better than multi-agent
reinforcement learning.

## Repository layout

| Path | Content |
|---|---|
| `revision_code/` | All scripts of the revised paper; `revision_code/README.md` gives the order of execution |
| `revision_code/outputs/` | Aggregate results (JSON) behind every table and figure of the revised paper |
| `revision_code/outputs/models2/` | Trained weights of all 50 learned policies (QMIX, MAPPO, MADDPG, two QMIX ablations; ten seeds each) and their training curves |
| `figures/` | All figures of the revised paper as PDF (vector graphics; dashboard screenshots at 600 dpi) |
| `legacy_submitted_version/` | Notebook of the risk classifier and dashboard (outputs cleared), and the superseded material of the submitted version |

## Reproducing the results

| What | How | Needs the ERP data |
|---|---|---|
| Tables of controllers, seeds and sensitivity | `python write_tables.py` (reads `outputs/sim3_summary.json`) | No |
| Figures of Sections 5.1-5.2 | `python fig_rq1.py` | No |
| Figures of Sections 5.3-5.5 (controllers, training curves, sensitivity) | `python plot_results.py` (reads `outputs/sim3_summary.json` and the training curves) | No |
| Segmentation figures (Appendix B) | `python fig_clusters.py` | Yes |
| Loss split, flag informativeness, influential items | values in `outputs/extra_analysis.json`, `outputs/influence_check.json` | No (to read); yes (to recompute) |
| Trained policies | load any `outputs/models2/*.pt` with PyTorch; network definitions in `evaluate3.py` | No |
| Simulator inputs, training, evaluation, decision value, out-of-year test | scripts in the order of `revision_code/README.md` | Yes |
| Risk classifier (Sections 4.2, 5.1; Appendix E) and dashboard | `legacy_submitted_version/ml-inventory-obsculence.ipynb` | Yes |

## Data

The ERP transaction data are proprietary to the participating firm and are not publicly available. Files that would
reveal SKU-level records (the simulator inputs, per-SKU evaluation results and action logs) are not included; the
scripts that produce them are, and the feature definitions are given in the paper's appendix.

## Requirements

Python 3.12 with numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, imbalanced-learn, torch, shap and matplotlib.

## License

MIT (see `LICENSE`).
