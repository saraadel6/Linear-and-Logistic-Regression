import ml_assignment1 as d
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1D Array
y_train = d.y_train['Emission Class'].to_numpy().ravel()
y_test = d.y_test['Emission Class'].to_numpy().ravel()

# SGDClassifier
Logistic_model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000)

Logistic_model.fit(d.X_train, y_train)

# Make predictions
y_predict = Logistic_model.predict(d.X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f"Model Accuracy: {accuracy:.2f}")

