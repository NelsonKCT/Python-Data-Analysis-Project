# Predicting Final Grades (G3)
# The following analysis aims to predict students' final grades (G3) using various features, such as demographics, family background, and study-related factors.
# Understanding what influences student performance can help us know how to improve educational outcomes.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Get file
file_id = '1ttLtLXzFzITD0R7KVgDqkIEEyZlZQSaQ'
download_url = f'https://drive.google.com/uc?id={file_id}'

df = pd.read_csv(download_url)

# Data Preprocessing
# Scale Numerical Features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Encode Categorical Features
# Binary mappings for two-category nominal variables
# One-Hot Encoding for nominal variables with more than two categories
def encode_categorical_variables(df):
    binary_mapping = {'no': 0, 'yes': 1}
    df['sex'] = df['sex'].map({'F': 0, 'M': 1})
    df['school'] = df['school'].map({'GP': 0, 'MS': 1})
    df['address'] = df['address'].map({'U': 0, 'R': 1})
    df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})
    df['Pstatus'] = df['Pstatus'].map({'A': 0, 'T': 1})
    binary_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    df[binary_columns] = df[binary_columns].replace(binary_mapping)

    df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)

    return df

# Displays the correlation of all features with G3
df = encode_categorical_variables(df)
correlations = df.corr()['G3'].sort_values(ascending=False)
print(correlations)

# Model Training
features = [
    'age', 'sex', 'studytime', 'failures', 'absences', 'goout',
    'freetime', 'famrel', 'health', 'Medu', 'Fedu', 'Dalc', 'Walc',
    'activities', 'romantic', 'internet'
]


# Prepare data
X = df[features]
y = df['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)


# Train model
model = RandomForestRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=10, n_estimators=500, random_state=50)
model.fit(X_train, y_train)

# Evaluation
# Evaluate model
y_pred = model.predict(X_test)

r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')

# Plot actual vs. predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs. Predicted Final Grades')
plt.show()

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame
feat_importances = pd.Series(importances, index=feature_names)
feat_importances = feat_importances.sort_values(ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
feat_importances.plot(kind='barh')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.show()

# Conclusion
# The Random Forest Regression model applied to predict students' final grades (G3) demonstrates moderate predictive capability, accounting for approximately 30.1% of the variance as indicated by the R-squared value. The scatter plot of actual versus predicted grades reveals a noticeable spread around the ideal prediction line, underscoring the model's limited accuracy in capturing all the factors influencing academic performance.

# The feature importance analysis highlights that the number of past class failures, absences, and mother's education level are the most significant predictors of a student's final grade. This suggests that a student's academic history and parental educational background are critical factors in their academic success. In contrast, features such as sex, participation in extracurricular activities, and internet access have a negligible impact on the model's predictions.