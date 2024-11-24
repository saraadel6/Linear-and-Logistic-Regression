import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# a) Load the "co2_emissions_data.csv" dataset.
data = pd.read_csv('co2_emissions_data.csv')
# print(data.head())

# b) i) check whether there are missing values
missing_values = data.isnull().sum()
# print(missing_values)

# b) ii) check whether numeric features have the same scale
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    print(f"{col}: Min = {data[col].min()}, Max = {data[col].max()}")

# b) iii) visualize a pairplot in which diagonal subplots are histograms
sns.pairplot(data, diag_kind='hist')

# b) iv) visualize a correlation heatmap between numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_data.corr(),annot=True,cmap='summer')
plt.title("Correlation Heatmap for Numeric Columns")

# c) i) the features and targets are separated
x = data.iloc[:, :-2]
y = data.iloc[:, -2:]
# print(x)
# print(y)

# c) ii) categorical features and targets are encoded
from sklearn.preprocessing import LabelEncoder
featureCols = x.select_dtypes(include=['object']).columns
for featureCol in featureCols:
  labelEncoder = LabelEncoder()
  x[featureCol] = labelEncoder.fit_transform(x[featureCol])

targetCols = y.select_dtypes(include=['object']).columns
for target_col in targetCols:
    target_encoder = LabelEncoder()
    y[target_col] = target_encoder.fit_transform(y[target_col])

# c) iii) the data is shuffled and split into training and testing sets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
data = shuffle(data)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print (X_train)
# print (X_test)
# print (y_train)
# print (y_test)
# len(X_train)
# len(X_test)
# len(y_train)
# len(y_test)

# c) iv) numeric features are scaled
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric] = scaler.transform(X_test[numeric])
# print(X_train.head())