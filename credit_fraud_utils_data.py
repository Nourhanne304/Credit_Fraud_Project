import pandas as pd
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.drop_duplicates()

    rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df.drop(['Time','Amount'], axis=1, inplace=True)

    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

if __name__ == "__main__":
    # تحميل البيانات
    X, y = preprocess_data(load_data("creditcard.csv"))

    # تعريف StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    all_results = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # تطبيق SMOTE فقط على بيانات التدريب
        print(f"\n[Fold {fold}] قبل SMOTE:", Counter(y_train))
        smote = SMOTE(random_state=1, k_neighbors=3)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"[Fold {fold}] بعد SMOTE:", Counter(y_train_res))

        # تدريب نموذج بسيط (مثال: Logistic Regression)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_res, y_train_res)

        # التنبؤ
        y_pred = model.predict(X_test)

        # التقييم
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"[Fold {fold}] Test Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

        all_results.append((acc, f1))
        fold += 1

    # متوسط الأداء
    avg_acc = sum(r[0] for r in all_results) / len(all_results)
    avg_f1 = sum(r[1] for r in all_results) / len(all_results)
    print(f"\nAverage Accuracy: {avg_acc:.4f}, Average F1-score: {avg_f1:.4f}")
