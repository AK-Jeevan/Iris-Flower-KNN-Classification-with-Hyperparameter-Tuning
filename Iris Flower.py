# This model classifies iris flowers into 3 species based on petal/sepal measurements using K-Nearest Neighbors (KNN) and Hyperparameter Tuning.

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load and clean data
data = pd.read_csv("C://Users//akjee//Documents//ML//iris_data.csv")
# data.drop_duplicates(inplace=True)
# data.dropna(inplace=True)
data_size = data.shape
print("Data Size:", data_size)
print(data.columns)

# Split features and target
X = data.iloc[:, :-1]  # Independent variables
y = data.iloc[:, -1]  # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(f"X Train shape is :{X_train.shape}")
print(f"X Test shape is :{X_test.shape}")
print(f"Y Train shape is :{y_train.shape}")
print(f"Y Test shape is :{y_test.shape}")

# Feature scaling (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning (to find optimal k) using GridSearchCV
param_grid = {
    'n_neighbors': np.arange(1, 30),
    'weights': ['uniform']
}

grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2 )
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
best_grid = grid_search.best_estimator_

# Predictions
y_pred = best_grid.predict(X_test_scaled)
print(y_pred.size)
print("Predictions:", y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_grid.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report and accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model Accuracy:", grid_search.score(X_test_scaled, y_test))

# Finding the best k value by plotting the graph between k values and accuracies
results = pd.DataFrame(grid_search.cv_results_)
k_values = param_grid['n_neighbors']
mean_scores = []
for k in k_values:
    # Get the best mean test score for each k (across all weights and metrics)
    mean_scores.append(results[results['param_n_neighbors'] == k]['mean_test_score'].max())

plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_scores, marker='o')
plt.title("K Values vs. Accuracy")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Best Accuracy for K")
plt.xticks(k_values)
plt.grid(True)
plt.show()