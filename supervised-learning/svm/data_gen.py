

import pandas as pd

def create_svm_dataset():
    """Create and save a sample binary classification dataset."""

    data = {
        "Feature1": [1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8],
        "Feature2": [2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7],
        "Label": [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
    }

    df = pd.DataFrame(data)
    df.to_csv("svm_classification_data.csv", index=False)

    print("Dataset saved successfully!")
    print(df)


create_svm_dataset()
