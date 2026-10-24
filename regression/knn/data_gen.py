
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_dataset():
    """Generate and save a synthetic regression dataset as a CSV file."""
    np.random.seed(42)

    X = 10 * np.random.rand(100)
    y = 2 * X + 5 + np.random.randn(100) * 2

    # Create a DataFrame
    df = pd.DataFrame({
        "X": X,
        "y": y
    })

    # Save CSV in the current working directory
    df.to_csv("knn_regression_data.csv", index=False)

    # Visualize the dataset
    plt.scatter(X, y)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Sample Dataset for KNN Regression")
    plt.show()

    print("Dataset saved to knn_regression_data.csv")
    print(df.head())
    print("Dataset shape:", df.shape)

generate_dataset()
