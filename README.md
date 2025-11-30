# ML_Fairness_Analysis

Project for CSE190 exploring gender fairness in salary prediction. We study whether an employee earns above the dataset median salary using gender as the sensitive attribute and test several fairness interventions.

- Data source: Kaggle salary survey combining job sites and government statistics (`Dataset/Salary.csv`).
- Artifact: full analysis and plots live in `Data_Analysis.ipynb`.
- Video walkthrough: see `video_url.txt` (Google Drive link).

## Setup
- Python 3.10+ with pandas, numpy, scikit-learn, matplotlib, seaborn, scipy.
- Quick install: `pip install pandas numpy scikit-learn matplotlib seaborn scipy`.

## What we did
- Clean/encode data: dropped `Job Title`, label-encoded `Country`/`Race`, binarized `Gender` (Female=0, Male=1) and `Salary` (above-median=1).
- Baseline inspection: 49.48% of rows are above-median salary; 53.88% of males vs 44.11% of females exceed the median, hinting at imbalance before modeling.
- Metrics: Balanced Accuracy for utility and per-group True Positive Rate (TPR) for fairness.

## Results
- Logistic baseline: accuracy 0.872; TPR (male 0.843, female 0.874).
- Dataset-based (resampling high earners): accuracy 0.874; TPR (male 0.855, female 0.858).
- Random Forest baseline: accuracy 0.866; TPR (male 0.902, female 0.848).
- In-processing weights on Random Forest: accuracy 0.854; TPR (male 0.821, female 0.824).
- Post-processing thresholds by group: accuracy 0.866; TPR (male 0.819, female 0.818).
- Convex fairness framework (from research paper): test accuracy 88.60%; TPR parity achieved (1.00, 1.00).

## Takeaways
- Gender gap appears in raw data and basic models.
- Simple resampling and group thresholds narrow TPR gaps with minimal accuracy loss.
- Weighted in-processing removes disparity but slightly lowers utility.
- The convex fairness approach delivered the best fairness without sacrificing accuracy in this dataset.
