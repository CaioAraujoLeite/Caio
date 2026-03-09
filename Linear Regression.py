import numpy as np


class LinearRegression:
    def __init__(self, lr = 0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0   
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            # Optional: print loss every 100 iterations
            if _ % 100 == 0:
                mse = np.mean(y_predicted - y) ** 2
                print(f"Iteration {_}: MSE = {mse:.4f}")
            
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
    
    
    
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your model
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Check accuracy (R² score or MSE)
mse = np.mean((predictions - y_test) ** 2)
print(f"MSE: {mse:.2f}")