# Machine Learning Analyses: Multimodal Classification Framework



This directory contains all machine-learning (ML) analyses conducted for the classification of diagnostic status and symptom state using multimodal data derived from spontaneous speech, neuropsychological testing, clinician-rated psychopathology, and structural neuroimaging. The codebase is designed to ensure strict methodological rigor, reproducibility, and comparability across modalities.



All analyses are implemented in Python using the PhotonAI framework and scikit-learn, and adhere to best-practice recommendations for clinical ML, including nested cross-validation, strict separation of training and test data, and harmonized preprocessing and model selection across experimental conditions.



---



## Scientific Rationale



Speech and language disturbances are transdiagnostic features of severe mental disorders. While previous studies have applied ML to individual modalities (e.g., speech, cognition, neuroimaging), direct comparisons across modalities are often confounded by differences in preprocessing, model space, or validation strategy.



The present framework was developed to address this limitation by enforcing \*\*identical analytical conditions across all modality-defined experiments\*\*, thereby enabling fair, interpretable comparisons of predictive utility and principled multimodal integration.



---



## Classification Tasks



The pipeline supports \*\*binary classification\*\* tasks, including:



- \*\*Diagnostic classification\*\*  

&nbsp; e.g., patient vs. healthy control



- \*\*Symptom-state classification\*\*  

&nbsp; e.g., acute vs. non-acute symptom status



Targets can be provided either as:

- native binary variables (0/1), or

- continuous clinical scores transformed into binary labels via thresholding (e.g., HAMD ≤ 6).



---



## Modalities and Feature Definitions



Feature sets are centrally defined and version-controlled via registry files (`src/features.py`, `src/features\_spaps.py`). Modalities include:



- \*\*Speech acoustics\*\*  

&nbsp; Low-level acoustic descriptors (e.g., prosody, spectral features, MFCCs) extracted using standardized openSMILE configurations.



- \*\*Transcript-based linguistic features\*\*  

&nbsp; Lexical, syntactic, semantic, and coherence-related measures derived from NLP pipelines, including embedding-based similarity indices.



- \*\*Neurocognition\*\*  

&nbsp; Standardized neuropsychological test scores spanning executive function, memory, attention, and processing speed.



- \*\*Clinician-rated psychopathology\*\*  

&nbsp; Established rating scales assessing positive, negative, affective, and global symptom severity.



- \*\*Structural neuroimaging (sMRI / connectivity)\*\*  

&nbsp; Cortical thickness, surface-based morphometry, subcortical volumes, and network-level connectivity metrics.



Each experiment differs \*\*only\*\* in the feature columns included; preprocessing, model space, and validation procedures are otherwise identical.



---



## Pipeline Architecture



All experiments are implemented using a shared, modular pipeline architecture:



\. \*\*Preprocessing\*\*

   - Optional Mean or median imputation (configurable)

&nbsp;  - Robust scaling (median and IQR normalization)

&nbsp;  - Variance thresholding

&nbsp;  - Principal component analysis (PCA; variance-retention based)



\. \*\*Model space\*\*

&nbsp;  - Regularized logistic regression

&nbsp;  - Support vector machines

&nbsp;  - Random forests

&nbsp;  - Gradient boosting / AdaBoost  

&nbsp;  Class imbalance is handled via class-weighted loss functions.



\. \*\*Model selection\*\*

&nbsp;  - Grid search optimization

&nbsp;  - Selection criterion: \*\*balanced accuracy\*\*



---



## Validation Strategy



To prevent information leakage and optimistic bias, all analyses employ \*\*nested cross-validation\*\*:



- \*\*Outer loop\*\*: unbiased performance estimation  

- \*\*Inner loop\*\*: hyperparameter optimization and model selection  



The number of folds is \*\*adaptively determined\*\* based on the minority class size to ensure valid stratified splits in imbalanced or subgroup-restricted analyses.



All preprocessing and optimization steps are confined strictly to the training partitions of each fold.



---



## Multimodal Integration Strategies



Two complementary integration approaches are supported:



- \*\*Early fusion\*\*  

&nbsp; Feature-level concatenation followed by dimensionality reduction and classification.



- \*\*Late fusion / stacking\*\*  

&nbsp; Modality-specific base models are trained in parallel, and their predicted probabilities are combined via a meta-classifier within the same nested CV framework.



This design enables principled comparison of unimodal, early-fusion, and late-fusion models under identical validation conditions.








