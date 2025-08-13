# Credit Card Fraud Detection

This project detects fraudulent credit card transactions using various machine learning models and sampling techniques to handle class imbalance.

## 📂 Project Structure
- **credit_fraud_utils_data.py**: Functions for loading, preprocessing, and sampling the data.  
- **main.py**: Script for training and evaluating models.  
- **creditcard.csv**: Dataset file (must be downloaded separately from Kaggle and placed in the same directory or provide its path).  

📥 **Dataset**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/code?datasetId=310&sortBy=voteCount&searchQuery=tsne)

## ⚙ Features
- Supports multiple models:  
  - Logistic Regression  
  - Random Forest  
  - SVM  
  - MLP (Multi-Layer Perceptron)  
  - Voting Classifier (ensemble of models)  
- Data preprocessing:  
  - Duplicate removal.  
  - Robust scaling for `Time` and `Amount` features.  
- Sampling options:  
  - Over-Sampling (SMOTE)  
  - Under-Sampling  

## 📦 Requirements
Install the dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn
Usage
Example commands:
bash
python main.py --data_path creditcard.csv --model logistic_regression
bash
python main.py --data_path creditcard.csv --model random_forest
Output Example
When running, the script prints accuracy and F1-score for train, validation, and test sets:

yaml
Copy
Edit
Training model: logistic_regression...
Train Accuracy: 0.9567, F1-score: 0.9325
Validation Accuracy: 0.9648, F1-score: 0.9425
Test Accuracy: 0.9683, F1-score: 0.9552
📊 Model Performance Comparison
Model	Train Accuracy	Train F1	Validation Accuracy	Validation F1	Test Accuracy	Test F1
Logistic Regression	0.9567	0.9325	0.9648	0.9425	0.9683	0.9552
Random Forest	1.0000	1.0000	0.9718	0.9524	0.9437	0.9175
MLP	1.0000	1.0000	0.9577	0.9318	0.9613	0.9447
Voting Classifier	0.9728	0.9577	0.9789	0.9647	0.9577	0.9394
SVM	0.9527	0.9238	0.9577	0.9286	0.9331	0.9036

Note: The above results are from experiments after applying sampling techniques (under/over sampling).

💡 Notes
Sampling method can be changed in load_and_split_data inside credit_fraud_utils_data.py.

Undersampling reduces dataset size and may affect generalization.

In experiments, MLP and Voting Classifier performed the best on the test set.

📜 License
This project is for educational and experimental purposes only.
