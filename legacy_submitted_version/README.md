# Submitted version (superseded)

This folder keeps the notebook, dashboard pages and aggregate outputs of the first submitted version of the paper.

**Still used by the revised paper:** the notebook `ml-inventory-obsculence.ipynb` contains the risk classifier
(XGBoost, LightGBM, Random Forest, BiLSTM and their ensemble; Sections 4.2 and 5.1 and Appendix E of the revised
paper), the segmentation and the dashboard.

**Superseded:** everything else in the notebook and in `pipeline_outputs_submitted.zip`. The following results of the
submitted version are withdrawn: the learned transition (surrogate) model and the rewards of the policies trained in
it; the positive episode rewards and the failure-rate, time-to-obsolescence and working-capital figures derived from
them; the time-to-inactivity forecast accuracy; and the significance tests based on three seeds. The revised results
are produced by the code in `../revision_code/`.

Per-SKU outputs (scored SKU lists, cluster assignments, the SKU table of the prediction page and the notebook's printed outputs) have been removed.
