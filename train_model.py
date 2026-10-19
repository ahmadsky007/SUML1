import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    print(" Step 1: Load and Explore the Dataset")
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"Number of samples:      {X.shape[0]}")
    print(f"Number of features:     {X.shape[1]}")
    print(f"Feature names:          {list(iris.feature_names)}")
    print(f"Target / Class names:   {list(str(name) for name in iris.target_names)}")
    print(f"First row of X (sample features): {X[0]}")
    print(f"First target label (y[0]):        {y[0]} ({iris.target_names[y[0]]})")

    print("\n Step 2: Create Training and Test Sets ")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Total dataset size: {len(X)} samples")
    print(f"X_train samples:    {len(X_train)} (80%)")
    print(f"X_test samples:     {len(X_test)} (20%)")

    print("\n--- Step 3: Choose and Train a Classifier ---")
    print("Selected Classifier: LogisticRegression")
    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)
    print("Model training completed using model.fit(X_train, y_train).")

    print("\n--- Step 4: Make Predictions ---")
    y_pred = model.predict(X_test)

    print("Sample comparison of Actual vs Predicted classes (first 10 test samples):")
    print(f"{'Index':<6} | {'Actual (y_test)':<18} | {'Predicted (y_pred)':<18} | {'Match'}")
    print("-" * 55)
    for i in range(min(10, len(y_test))):
        actual_name = iris.target_names[y_test[i]]
        pred_name = iris.target_names[y_pred[i]]
        match = "✓" if y_test[i] == y_pred[i] else "✗"
        print(f"{i:<6} | {actual_name:<18} | {pred_name:<18} | {match}")

    print("\n--- Step 5: Evaluate the Model ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    correct_predictions = int(cm.diagonal().sum())
    total_predictions = len(y_test)
    incorrect_predictions = total_predictions - correct_predictions

    print(f"\nBreakdown:")
    print(f"• Correct predictions:   {correct_predictions} / {total_predictions}")
    print(f"• Incorrect predictions: {incorrect_predictions} / {total_predictions}")
    if incorrect_predictions == 0:
        print("• Confused classes:      None (all test instances were predicted correctly)")

    print("\n--- Step 6: Save the Trained Model ---")
    model_filename = "model.joblib"
    joblib.dump(model, model_filename)

    if os.path.exists(model_filename):
        print(f"Verification: '{model_filename}' created successfully ({os.path.getsize(model_filename)} bytes).")
    else:
        print(f"Error: '{model_filename}' could not be found.")

    print("\n--- Step 7: Load and Reuse the Model ---")
    loaded_model = joblib.load(model_filename)
    print(f"Loaded model from '{model_filename}'.")

    loaded_pred = loaded_model.predict(X_test)
    loaded_accuracy = accuracy_score(y_test, loaded_pred)

    print(f"Original Model Accuracy: {accuracy:.4f}")
    print(f"Loaded Model Accuracy:   {loaded_accuracy:.4f}")

    if loaded_accuracy == accuracy and (y_pred == loaded_pred).all():
        print("Comparison: The loaded model produces identical predictions to the original trained model!")
    else:
        print("Comparison: Discrepancy detected between original and loaded model predictions.")


if __name__ == "__main__":
    main()
