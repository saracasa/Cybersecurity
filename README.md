# PROJECT 7 - Privacy-preserving synthetic healthcare data generation

## Research question
Can synthetic healthcare data generated for cybersecurity protection maintain sufficient statistical fidelity and research utility to replace real patient data in medical research and machine learning applications?

## Introduction
Healthcare data breaches are increasingly common and devastating, yet researchers need access to realistic datasets for medical AI development and clinical trials. Synthetic data generation offers a cybersecurity solution: if breached, synthetic datasets contain no real patient information, minimizing legal liability and patient harm. However, synthetic data is only valuable if it preserves the properties necessary for valid research. This project evaluates the fundamental trade-off: is it possible to generate synthetic healthcare data that is simultaneously secure against privacy attacks and useful for medical research?

Specifically, the study compares the application of three privacy guarantee levels (no privacy, moderate privacy and strong privacy) to identify which techniques best balance privacy protection with data utility.

## Replicability
The project was developed to ensure maximum transparency and to be fully replicable by running the notebook `progettoCybersecurity.ipynb`. The entire workflow is managed within the `progettoCybersecurity.ipynb` notebook, which offers the user two distinct operational paths based on their computational needs:

- **Full reproduction of the experiment (from scratch):** the notebook downloads the original dataset from Kaggle, performs preprocessing, trains the CTGAN and DP-CTGAN models, and generates new synthetic datasets usable for all subsequent phases.
- **Rapid analysis and consultation of our experiment results (loading pre-existing data):** to facilitate the project review without the need for high computational resources or long waiting times, it is possible to proceed in two ways:
  - **Static consultation:** The notebook is delivered with the most important cell outputs already rendered. It is therefore possible to analyze graphs, quality metrics, and MIA attack results directly from the pre-executed version.
  - **Partial execution (data loading):** It is possible to skip the training phases by loading the already generated synthetic datasets. In this way, the user can instantly re-run only the analysis cells, verifying the validity of the results in real-time.

## Notebook structure
The project is divided into the following main phases:

1. **Installation of libraries and imports necessary for the project.**
2. **Loading and preprocessing:** importation of the "Diabetes Health Indicators" dataset from Kaggle (https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data?select=diabetes_binary_health_indicators_BRFSS2015.csv) and its subdivision into train and holdout sets, or the use of predefined datasets to ensure exact reproducibility of the project.
3. **Synthetic data generation:**
   - For the dataset without privacy guarantees, the `CTGANSynthesizer` model (`Synthetic Data Vault` (SDV) v1.29.1) was employed.
   - For the protected datasets, the `DP-CTGAN` model from the `SmartNoise-Synth` library was used, which extends CTGAN by integrating Differential Privacy principles. The level of privacy guaranteed by the generative model is determined by the epsilon parameter ($\varepsilon$): high values of $\varepsilon$ (> 2.0) correspond to low privacy protection, while lower values (0.1 – 2.0) indicate high protection. In our study, we set $\varepsilon$ = 4.0 for moderate privacy and $\varepsilon$ = 2.0 for strong privacy.
4. **Evaluation of synthetic data similarity compared to real data:**
   - Analysis of synthetic data quality through the _Diagnostic Report_ and _Quality Score_ provided by SDV. The _Diagnostic Report_ evaluates whether the synthetic dataset is structurally valid, i.e., if it respects the formal and semantic constraints defined by the original data schema. The _Quality Score_ provides a quantitative measure of the statistical fidelity of the synthetic data compared to the real dataset.
   - Evaluation of statistical similarity by comparing means, standard deviations, variable distributions, correlation matrices, and Frobenius distance between synthetic and real data.
5. **Evaluation of utility for medical research:**

   The goal of this phase is to verify if the clinical knowledge extracted from the synthetic dataset is transferable to real-world contexts. To test this research utility, we adopted the TSTR (Train on Synthetic, Test on Real) protocol: models are trained exclusively on synthetic data and their predictive capacity is measured on real data never seen by the generator (holdout set).

   To ensure a robust evaluation, four different Machine Learning architectures were employed, chosen for their different complexity and algorithmic nature:
     - Logistic Regression
     - Random Forest
     - XGBoost
     - MLP (Multi-Layer Perceptron)
    
    The effectiveness of the models was quantified via ROC-AUC. This metric was chosen because:
     - **Threshold independence:** It evaluates the model's ability to distinguish between classes (e.g., diabetic vs. non-diabetic) at all possible confidence levels.
     - **Robustness to imbalance:** It is more reliable than simple accuracy in medical datasets where class distribution may be imbalanced.
7. **Privacy evaluation:**

     A _Membership Inference Attack_ was implemented to measure information loss. The attack was formulated as a binary classification problem. The attacker is an XGBoost classifier trained on a dataset composed exclusively of real data labeled as members, coming from the generator's training set, and non-members, coming from a never-before-seen holdout dataset. The attacker does not have direct access to the generator but only observes the considered synthetic dataset and, based on this, builds features that measure how well each real record is represented by the synthetic data.
    - The first feature is the **distance to the k-nearest neighbors**, which measures how close a real record is to the k nearest synthetic samples.
    - The second is **local density**, which evaluates how much synthetic data mass is concentrated around that record.
    - Finally, the **reconstruction error** measures how well the relationships between the features of the real record are consistent with those learned from the synthetic dataset.
    
    Attack performance is evaluated via ROC-AUC, which is independent of the decision threshold. A ROC-AUC equal to 0.5 indicates a random attack, while higher values represent a loss of privacy.

10. **Privacy-Utility trade-off analysis:**

    We related two metrics: on the horizontal axis, clinical utility, measured via the ROC-AUC of the logistic regression model (which is the most robust among those analyzed and the ROC-AUC best reflects the actual utility of the model), and on the vertical axis, the privacy risk, measured by the ROC-AUC of the attack.

    To guide the interpretation, we defined two strategic thresholds:
    - A **threshold of acceptable utility** at 0.70, below which the model would not be clinically relevant;
    - An **optimal privacy threshold** at 0.50, which represents the random level for the attack: values close to 0.50 indicate complete anonymity protection.

    Consequently, the goal is to identify the model located in the green area of the graph, where clinical utility remains valid but the risk of privacy loss is minimized.
