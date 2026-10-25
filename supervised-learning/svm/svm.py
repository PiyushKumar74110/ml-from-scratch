
import numpy as np
import pandas as pd


def hinge_loss(y_true, y_pred):
    """Calculate hinge loss for one training sample."""

    margin = y_true * y_pred
    return max(0.0, 1.0 - margin)


def decision_score(x, weights, bias):
    """Calculate the raw decision score for one sample."""

    return np.dot(x, weights) + bias


def calculate_gradients(x, y_true, weights, bias, C):
    """Calculate per-sample gradients for the SVM objective."""

    y_pred = decision_score(x, weights, bias)
    margin = y_true * y_pred


    if margin >= 1:
        dw = weights.copy()
        db = 0.0
    else:
        dw = weights - C * y_true * x
        db = -C * y_true

    return dw, db


def train_svm(X, y, learning_rate=0.001, epochs=1000, C=1.0):
    """Train a linear SVM using stochastic gradient descent."""

    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        for i in range(n_samples):
            dw, db = calculate_gradients(
                X[i], y[i], weights, bias, C
            )

            # Update model parameters
            weights -= learning_rate * dw
            bias -= learning_rate * db

    return weights, bias


def predict(X, weights, bias):
    """Predict class labels for one or more samples."""

    scores = np.dot(X, weights) + bias
    return np.where(scores >= 0, 1, -1)


def accuracy_score(y_true, y_pred):
    """Calculate classification accuracy as a fraction."""

    return np.mean(y_true == y_pred)


def main():
    """Load the dataset, train the SVM, and evaluate predictions."""

    df = pd.read_csv(
        "supervised-learning/svm/svm_classification_data.csv"
    )

    X = df[["Feature1", "Feature2"]].to_numpy(dtype=float)
    y = df["Label"].to_numpy(dtype=int)

    if not np.isin(y, [-1, 1]).all():
        raise ValueError("SVM labels must be -1 or +1.")

    weights, bias = train_svm(
        X,
        y,
        learning_rate=0.001,
        epochs=1000,
        C=1.0,
    )

    y_pred = predict(X, weights, bias)
    accuracy = accuracy_score(y, y_pred)

    print("Learned weights:", weights)
    print("Learned bias:", bias)
    print("Predictions:", y_pred)
    print("Actual labels:", y)
    print(f"Training accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
