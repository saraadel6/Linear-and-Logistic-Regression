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
# print(data.describe())

# b) iii) visualize a pairplot in which diagonal subplots are histograms

sns.pairplot(data, diag_kind='hist')
# if __name__ == "__main__":
#   plt.show()

# b) iv) visualize a correlation heatmap between numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_data.corr(),annot=True,cmap='summer')
plt.title("Correlation Heatmap for Numeric Columns")
# if __name__ == "__main__":
  # plt.show()

# any_missing = data.isnull().values.any()
# print("Are there any missing values?", any_missing)

# numeric_data = data.select_dtypes(include=['float64', 'int64'])
# print(numeric_data.describe())
# for column in numeric_data.columns:
#     col_range = numeric_data[column].max() - numeric_data[column].min()
#     col_std = numeric_data[column].std()
#     print(f"{column}: Range = {col_range}, Standard Deviation = {col_std}")

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

# len(X_train)
# len(X_test)
# len(y_train)
# len(y_test)

#extracting features for linear reggression "Engine Size"/ "Fuel Consumption City"
# y = data['CO2 Emissions(g/km)']

# print(y[1])
# print(len(y_train))

# c) iv) numeric features are scaled
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric] = scaler.transform(X_test[numeric])
# print(X_train.head())

# print(X_train['Engine Size(L)'], X_train['Fuel Consumption City (L/100 km)']
#       ,y_train['CO2 Emissions(g/km)'])
