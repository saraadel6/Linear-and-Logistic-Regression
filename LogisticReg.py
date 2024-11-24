import ml_assignment1 as d
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# selected features
x_train_features = d.X_train[['Engine Size(L)', 'Fuel Consumption City (L/100 km)']]
x_test_features = d.X_test[['Engine Size(L)', 'Fuel Consumption City (L/100 km)']]

# convert the target to 1d array
y_train = d.y_train['Emission Class'].to_numpy().ravel()
y_test = d.y_test['Emission Class'].to_numpy().ravel()

# SGDClassifier
Logistic_model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000)

# Fit
Logistic_model.fit(x_train_features, y_train)

# prediction
y_predict = Logistic_model.predict(x_test_features)

# accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f"Model Accuracy: {accuracy:.2f}")