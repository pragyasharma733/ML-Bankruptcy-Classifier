Bankruptcy Prediction with Machine Learning

Predicting corporate bankruptcy from financial ratios using classical machine learning, with a focus on highly imbalanced data and recall. Final project for CFRM 521 (Machine Learning for Finance), University of Washington, 2022.

Team: Pragya Sharma and Rohan Tiwari

Why this problem

Bankruptcy prediction gives creditors, investors, and policy makers an early warning so they can act before losses pile up. Missing a firm that goes bankrupt is far more costly than a false alarm, so models were judged primarily on recall for the bankrupt class, alongside ROC AUC.

Data
Source: Taiwanese Bankruptcy Prediction dataset (UCI ML Repository; also on Kaggle), collected from the Taiwan Economic Journal, 1999–2009
Size: 6,819 firms, 95 financial features (ratios such as ROA, debt ratio, and current ratio), and 1 target (Bankrupt?)
Imbalance: only 220 firms (3.2%) went bankrupt
Approach
Exploratory analysis: class distribution, correlation of features with the target, and relationships among the most correlated features
Preprocessing: stratified train/test split (80/20), Min-Max scaling, and checks for missing and categorical values
Handling imbalance: class_weight='balanced' and recall-focused evaluation, since accuracy is misleading at 3% prevalence
Modeling: hyperparameter tuning with RandomizedSearchCV and GridSearchCV, plus learning curves to diagnose overfitting
Feature selection: random-forest feature importance, keeping the features that make up 95% of cumulative importance, then retraining models on that subset
Dimensionality reduction: PCA (16 components capture 95% of the variance)
Ensembles: bagging, AdaBoost, gradient boosting, Extra-Trees, and hard and soft voting classifiers
Results (test set)
Model	Accuracy	ROC AUC	Precision	Recall	F1
Logistic Regression	0.859	0.934	0.169	0.864	0.283
SVM (RBF)	0.857	0.937	0.167	0.864	0.280
SVM + PCA	0.844	0.929	0.152	0.841	0.258
Random Forest	0.916	0.938	0.243	0.750	0.367
AdaBoost	0.930	0.909	0.250	0.591	0.351
Gradient Boosting	0.900	0.935	0.211	0.773	0.332
Logistic Regression (selected features)	0.868	0.936	0.179	0.864	0.297
SVM (selected features)	0.856	0.938	0.167	0.864	0.279
Extra-Trees (selected features)	0.922	0.934	0.269	0.818	0.404
Bagging	0.972	0.923	1.000	0.136	0.240
k-NN	0.970	0.822	0.800	0.091	0.163
Soft Voting	0.940	0.939	0.310	0.705	0.431
Key takeaways
The best model for this problem is an SVM with an RBF kernel, with or without feature selection. It catches 86% of bankrupt firms, has a high ROC AUC, and shows little overfitting. This matches the best model reported by Liang et al. (2016).
Feature selection kept performance the same with fewer features. Logistic regression on the selected features slightly improved accuracy and F1 at the same recall.
High accuracy can be misleading. Bagging and k-NN reach about 97% accuracy but miss most bankrupt firms (recall of 14% and 9%).
Soft voting gives the best balance of precision and recall (highest F1 and ROC AUC), at the cost of some recall.
Contributions
Pragya Sharma: exploratory data analysis and visualization; feature selection using random forests; logistic regression and SVM on the selected features; Extra-Trees (with and without feature selection); bagging and pasting; k-NN; hard and soft voting classifiers
Rohan Tiwari: preprocessing (class imbalance, missing values, scaling) and evaluation utilities (metrics, ROC and learning curves); logistic regression; SVM; PCA and SVM with PCA; random forest; AdaBoost; gradient boosting
Both: summary of results and conclusions
Repository contents
File	Description
CFRM521_FinalProject - 06-04-2022.ipynb	Final notebook with the full analysis
CFRM521_FinalProject_06_04_2022-pragya-version.ipynb	Pragya's working version
CFRM521_FinalProject-OLD.ipynb	Earlier draft
data.csv	Dataset

Tools: Python, scikit-learn, pandas, NumPy, Matplotlib, Google Colab

References
Liang, D., Lu, C.-C., Tsai, C.-F., and Shih, G.-A. (2016). Financial ratios and corporate governance indicators in bankruptcy prediction: A comprehensive study. European Journal of Operational Research, 252(2).
Wang, H. and Liu, X. (2021). Undersampling bankruptcy prediction: Taiwan bankruptcy data. PLOS ONE. doi:10.1371/journal.pone.0254030
