import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the training dataset
train_df = pd.read_csv("../data/gender-voice-train.csv")

# Extract features and target variable
x_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]

# Initialize and train the Random Forest model
rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=50)
rf_clf.fit(x_train, y_train)

# Save the trained model
import joblib
joblib.dump(rf_clf, '../models/random_forest_model.pkl')
print("Model saved successfully.")