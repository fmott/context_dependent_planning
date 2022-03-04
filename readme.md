## Forward planning driven by context-dependent conflict processing in anterior cingulate cortex - Analysis code and datasets

**Cite:** Ott, F., Legler, E., & Kiebel, S. J. (2021). Forward planning driven by context-dependent conflict processing in anterior cingulate cortex. bioRxiv 2021. https://biorxiv.org/content/10.1101/2021.07.19.452905 

The repository contains behavioural raw data, fMRI statistical maps and analysis scripts to reproduce the result of the study. 

### Description

* #### Data

  * **behaviour:** Raw behavioural data data of all 40 participants.
  * **fmri:**  fMRI statistical maps underlying Figure 5, Figure 6 and Tables S3-S6. Contains individual contrast images for all 40 participants and group level unthresholded T-maps. Also contains masks of the dorsal anterior cingulate cortex (dACC) and the dorsolateral prefrontal cortex (dlPFC), used for small volume correction. 
  * **model:** Contains files with posterior samples and summary statistics of the winning simple strategy model (SM), planning strategy model (PM), hybrid strategy model (HM) and the reaction time analysis. Contains posterior predictive simulations for PM, SM and HM. Also contains the leave-one-out information criterion (LOOIC) and standard error (SE) for the PM, SM and HM and for alternative feature-contingent planning models, heuristic models and hybrid models.

* #### Code

  * **generate_figure2.py:** Generates Figure 2. 
  * **generate_figure3.py:** Generates Figure 3.
  * **generate_figure4.py:** Generates Figure 4.
  * **generate_S1_figure.py:** Generates S2 Figure.
  * **generate_S2_figure.py:** Generates S2 Figure.
  * **generate_S3_figure.py:** Generates S2 Figure.
  * **generate_S5_figure.py:** Generates S2 Figure.
  * **generate_S6_figure.py:** Generates S2 Figure.
  * **generate_S2_table.R:** Logistic mixed-effects regression testing for sequence effects of segment type underlying S2 table.
  * **notebook_fit_choice_models.ipynb:** Jupyter notebook implementing model fitting and computation of LOOIC for the three models PM, SM and HM. 
  * **notebook_fit_RT_model.ipynb:** Jupyter notebook implementing hierarchical Bayesian linear regression of response times against conflict and context.
  * **notebook_parameter_recovery.ipynb:** Jupyter notebook implementing parameter recovery analysis for model HM.
  * **notebook_model_differentiability_analysis.ipynb:** Jupyter notebook implementing a model differentiability analysis.
  * **ara_backward_induction.py:** Function used to compute expected long-term values via backward induction.

