import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('data.csv')

def loss_function(m, b, points):
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        total_error += (y-(m*x + b))**2

    return total_error/ float(len(points))



def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2/n) * x * (y-(m_now * x + b_now))
        b_gradient += -(2/n) * (y-(m_now * x + b_now))

    m = m_now - m_gradient*L
    b = b_now - b_gradient*L

    return m , b


# Train-Test Split
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))


m = 0
b = 0
L = 0.0001
epochs = 3000

for i in range(epochs):

    
    m, b = gradient_descent(m, b, train_data, L)

    # Calculate loss
    loss = loss_function(
        m,
        b,
        train_data
    )

    if i % 50 == 0:
            print(f"Epoch: {i} || Loss : {loss}")
    

print("m =", m)
print("b =", b)


# Predictions
predictions = m * test_data.studytime + b

# R² Score
actual = test_data.score

ss_res = ((actual - predictions) ** 2).sum()
ss_tot = ((actual - actual.mean()) ** 2).sum()

r2 = 1 - (ss_res / ss_tot)


print("\nTest Results")
print("R² Score:", r2)
print("R² Score (%):", r2 * 100)


# RMSE
mse = ((actual - predictions) ** 2).mean()
rmse = mse ** 0.5

print("RMSE:", rmse)

# Plot
plt.scatter(train_data.studytime, train_data.score, color="black", label = "Training Data")
plt.scatter(test_data.studytime, test_data.score, label = "Testing Data")

plt.plot(
    list(range(20, 80)),
    [m*x + b for x in range(20, 80)],
    color="red"
)

plt.show(block=False)
plt.pause(5)
plt.close()
    
