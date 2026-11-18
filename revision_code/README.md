# Code and results for the revised manuscript

Python 3.12 with numpy, pandas, scipy, scikit-learn, xgboost, imbalanced-learn, torch and matplotlib.
Run from this folder. The ERP data are proprietary; scripts expect `../data 2024/master_cleaned.csv`
and `../data 2025/` (2025 workbook) as in the authors' project.

| Order | Script | Produces | Paper |
|---|---|---|---|
| 1 | `shap_share.py` | `outputs/shap_share.json` | Sec. 5.1 |
| 2 | `decision_value.py`, `decision_value2.py`, `dv_concentration.py` | `decision_value*.json`, `dv_concentration.json` | Sec. 4.3, 5.2, Appendix C |
| 3 | `prep2.py` | `sim2_inputs.pkl`, `sim2_phat_check.json` (SKU traces; out-of-fold risk estimates) | Sec. 4.4, Appendix D |
| 4 | `replay_check.py` | `replay_check.json` (B0 reproduces the recorded November stock) | Sec. 4.4 |
| 5 | `planner_select.py` | `planner_selection_oof.json` (demand estimate of B5, validation SKUs) | Appendix D |
| 6 | `rl2_train.py ALGO SEED 6000 [both|dynamic] per_sku [auto|no]` (`run_maddpg.sh` for MADDPG) | `outputs/models2/*.pt` (50 policies) | Sec. 4.5 |
| 7 | `evaluate3.py SETTING` (seven settings; `run_eval.sh`) | `sim3_eval_*.csv`, `sim3_flags_*.json`, `sim3_actions_base.pkl`, `guard_choice.json`, `classical_choice.json` | Sec. 5.3-5.5 |
| 8 | `analyze3.py`, `plot_results.py` | `sim3_summary.json`; Figs. 4-6 (`plot_results.py` needs only the published summary and training curves) | Sec. 5.3-5.5 |
| 9 | `analyze_extra.py` | `extra_analysis.json` (loss by item type; flag AUC and blocked clearance) | Sec. 5.3-5.4 |
| 9b | `influence_check.py` | `influence_check.json` (opening-stock value; influential-SKU and leave-one-out checks) | Sec. 5.3, Appendix D |
| 10 | `write_tables.py` | LaTeX rows of the controller, seed and sensitivity tables | Sec. 5.3-5.5 |
| 11 | `out_of_year.py` | `out_of_year.json` | Sec. 5.6 |
| 12 | `fig_rq1.py`, `fig_clusters.py`, `wf_preview.py` | figures of Sec. 4-5.2 and Appendix B | |

`sim2.py` is the ledger simulator, `controllers2.py` the rule-based, classical and planning controllers,
`hindsight.py` the exact hindsight optimum, `env2.py` the environment wrapper used for training.
`forecast_check.py` reproduces the accuracy check of the submitted time-to-inactivity forecast cited in the response letter.
`wf_preview.py` compiles the study-design diagram (Fig. 1) as a standalone PDF.

## Files not published

The ERP data and the files derived from them at SKU level are proprietary and are not included: `outputs/sim2_inputs.pkl` (written by `prep2.py`), `outputs/sim3_eval_*.csv` and `outputs/sim3_actions_base.pkl` (written by `evaluate3.py`). Every aggregate result built from them is in `outputs/`.
