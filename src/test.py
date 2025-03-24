import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Load the test dataset
test_df = pd.read_csv("../data/gender-voice-test.csv")

# Extract features and target variable
x_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]

# Load the trained model
rf_clf = joblib.load('../models/random_forest_model.pkl')

# Evaluate the model
score = rf_clf.score(x_test, y_test)
y_predicted = rf_clf.predict(x_test)

print(f"Model Accuracy: {score}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predicted))
print("Classification Report:")
print(classification_report(y_test, y_predicted))