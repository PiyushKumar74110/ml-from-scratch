import math
import pandas as pd

df = pd.read_csv("A:/MACHINE LEARNING/ml-from-scratch/regression/knn/knn_regression_data.csv")

print(df.head(5))



def euclidean_distance(x1, x2):
    """Calculate the Euclidean distance between two data points."""

    diff = x1 - x2

    diff_sq = diff**2

    return math.sqrt(diff_sq)



X  = df["X"].values
y = df["y"].values

x_new = 3.74

distances = []

for x_train in X:
    dist = euclidean_distance(x_train, x_new)
    distances.append(dist)



# define the number of neighbours
k = 5

# Sort training indices by their distances

indices = list(range(len(distances)))

indices.sort(key = lambda i : distances[i])

nearest_indices = indices[:k]

nearest_targets = []

for i in nearest_indices:
    nearest_targets.append(y[i])

total = 0

for target in nearest_targets:
    total += target


y_pred = total/k

print(y_pred)





