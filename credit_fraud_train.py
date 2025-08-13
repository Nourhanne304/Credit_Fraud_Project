import argparse
from credit_fraud_utils_data import load_and_split_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
def get_model(name):
    cnter = Counter(y_train)
    ir = cnter[0] / cnter[1]
    rf = RandomForestClassifier(n_estimators=100)
    lr = LogisticRegression(solver='lbfgs', class_weight={0:1,1:ir})

    models = {
        "logistic_regression": lr,
        "random_forest": rf,
        "svm": SVC(probability=True),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "voting": VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
    }

    if name.lower() not in models:
        raise ValueError(f"Model '{name}' not supported. Choose from: {list(models.keys())}")
    return models[name.lower()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to train: logistic_regression, random_forest, svm, mlp, voting")
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(args.data_path)

    model = get_model(args.model)

    print(f"Training model: {args.model}...")
    model.fit(X_train, y_train)

    # Predict on train, val, test
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Calculate F1-score
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"Train Accuracy: {train_acc:.4f}, F1-score: {train_f1:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")
