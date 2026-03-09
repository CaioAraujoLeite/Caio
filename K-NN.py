import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))    

def KNN(X_train, y_train, X_test, k):
    predictions = []
    for i in X_test:
        distances = []
        for j in X_train:
            dist = euclidean_distance(i, j)
            distances.append(dist)
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[idx] for idx in k_indices]
        predictions.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
    return predictions





predictions = KNN(X_train, y_train, X_test, k=3)
print("Predictions:", predictions)
print("Actual:", list(y_test))
accuracy = sum(p == a for p, a in zip(predictions, y_test)) / len(y_test)
print(f"Accuracy: {accuracy * 100:.1f}%")