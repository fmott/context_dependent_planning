## Forward planning driven by context-dependent conflict processing in anterior cingulate cortex - Analysis code and datasets

Contains behavioural raw data, fMRI statistical maps and analysis scripts to reproduce the result of the study. 

### Description

* #### Data

  * **behaviour:** Raw behavioural data data of all 40 participants.
  * **fmri:**  fMRI statistical maps underlying Figure 4, Figure 5 and Tables S2-S5. Contains individual contrast images for all 40 participants and group level unthresholded T-maps. Also contains masks of the dorsal anterior cingulate cortex (dACC) and the dorsolateral prefrontal cortex (dlPFC), used for small volume correction. 
  * **model:** Contains files with posterior samples and summary statistics of the three computational models Planning (PM), Simple (SM) and Hybrid (HM) and the reaction time analysis. Contains posterior predictive simulations for HM. Also contains the leave-one-out information criterion (LOOIC) for the PM, SM and HM used for model comparison.

* #### Code

  * **generate_figure2.py:** Generates Figure 2. 
  * **generate_figure3.py:** Generates Figure 3.
  * **generate_S2_figure.py:** Generates S2 Figure.
  * **generate_S1_table.R:** Logistic regression of choice against task features underlying S1 table.
  * **fit_choice_models.ipynb:** Jupyter notebook implementing model fitting and computation of LOOIC for the three models PM, SM and HM. 
  * **fit_RT_model.ipynb:** Jupyter notebook implementing hierarchical Bayesian linear regression of response times against conflict and context.
  * **parameter_recovery.ipynb:** Jupyter notebook implementing parameter recovery analysis for model HM.
  * **backward_induction.py:** Function used to compute expected long-term values via backward induction.
  * **simulate_agents.py:** Simulate behavior of agents using a planning, simple, hybrid or random strategy. 

