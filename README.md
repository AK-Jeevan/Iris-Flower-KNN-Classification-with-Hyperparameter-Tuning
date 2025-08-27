# Iris KNN Classification with Hyperparameter Tuning

This project applies the **K-Nearest Neighbors (KNN)** algorithm to classify iris flowers into three species: *Setosa*, *Versicolor*, and *Virginica*. The model is trained on the classic **Iris dataset** and optimized using hyperparameter tuning techniques.

## 📊 Dataset
The dataset contains 150 samples with the following features:
- `sepal length`
- `sepal width`
- `petal length`
- `petal width`
- `species` (target)

## ⚙️ Model
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Hyperparameter Tuning**: Grid Search with cross-validation to find the optimal value of `k`
- **Library**: scikit-learn

## 📈 Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

## 🚀 How to Run

### Prerequisites
pip install pandas numpy scikit-learn matplotlib seaborn

Clone the repository:
git clone https://github.com/AK-Jeevan/Iris-KNN-Classification-with-Hyperparameter-Tuning.git
cd Iris-KNN-Classification-with-Hyperparameter-Tuning

Run the script:
python iris_knn_classifier.py

📂 Files
iris_data.csv: Dataset
iris_knn_classifier.py: Model training and evaluation
README.md: Project overview

📌 Highlights
Simple and interpretable classification model
Hyperparameter tuning for optimal performance
Visualizations of decision boundaries and confusion matrix

📜 License
This project is open-source under the MIT License.
