def main():
    csv_path = "heart.csv"
    
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA

    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    df = pd.read_csv(csv_path)

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\n--- First 15 rows ---")
    print(df.head(15).to_string(index=False))

    print("\n--- Info ---")
    df.info()

    print("\n--- Missing values ---")
    print(df.isna().sum())

    print("\n--- Target distribution (before encoding) ---")
    print(df["Heart Disease"].value_counts())

    plt.figure(figsize=(12, 8))
    df.drop(columns=["Heart Disease"]).hist(bins=20, figsize=(14, 10))
    plt.suptitle("Feature Histograms", y=1.02)
    plt.show()

    le = LabelEncoder()
    df["Heart Disease"] = le.fit_transform(df["Heart Disease"])  

    X = df.drop("Heart Disease", axis=1)
    y = df["Heart Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y 
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=42)  
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("\n--- PCA Info ---")
    print("Original features:", X_train.shape[1])
    print("PCA components kept:", pca.n_components_)
    print("Explained variance ratio sum:", round(pca.explained_variance_ratio_.sum(), 4))

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
    plt.axhline(0.95, linestyle="--")
    plt.title("Cumulative Explained Variance (PCA)")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.show()


    def train_eval_plot(model_name, model, Xtr, Xte, y_train, y_test, class_names):
        model.fit(Xtr, y_train)

        y_pred = model.predict(Xte)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n==================== {model_name} ====================")
        print(f"Accuracy: {acc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        return acc

    accuracies = {}

    nb = GaussianNB()
    accuracies["PCA + Naive Bayes"] = train_eval_plot(
        "PCA + Naive Bayes", nb, X_train_pca, X_test_pca, y_train, y_test, le.classes_
    )

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    accuracies["PCA + SVM (RBF)"] = train_eval_plot(
        "PCA + SVM (RBF)", svm, X_train_pca, X_test_pca, y_train, y_test, le.classes_
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    accuracies["PCA + Random Forest"] = train_eval_plot(
        "PCA + Random Forest", rf, X_train_pca, X_test_pca, y_train, y_test, le.classes_
    )

    results_df = pd.DataFrame(
        [{"Model": k, "Accuracy": v} for k, v in accuracies.items()]
    ).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    print("\n=== Accuracy Comparison Table ===")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")

    for i, v in enumerate(results_df["Accuracy"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
