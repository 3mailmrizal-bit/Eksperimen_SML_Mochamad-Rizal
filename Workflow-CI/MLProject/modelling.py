import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # 1. PENGATURAN PARAMETER VIA CLI
    parser = argparse.ArgumentParser(description="Workflow CI - Training RandomForest")
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah tree")
    parser.add_argument("--max_depth", type=int, default=10, help="Kedalaman tree")
    args = parser.parse_args()

    # 2. PENYIAPAN DATA
    # Pastikan file CSS_preprocessing.csv berada di folder yang sama
    df_model = pd.read_csv("CSS_preprocessing.csv")
    X = df_model.drop(columns=['Credit_Mix'])
    y = df_model['Credit_Mix']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. TRAINING MODEL
    print(f"🌲 Training: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. EVALUASI
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # 5. MLFLOW LOGGING (Murni Tanpa DagsHub)
    mlflow.set_experiment("Workflow_CI_Experiment")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(sk_model=model, name="ci_model")
        
        print("✅ Sukses: Model dan metrik tercatat di MLflow!")

if __name__ == "__main__":
    main()