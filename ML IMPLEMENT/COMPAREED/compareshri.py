# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data into a pandas DataFrame
data = {
    'Model_Name': ['Bajaj Avenger Cruise 220', 'Royal Enfield Classic 350cc', 'Hyosung GT250R',
                   'Bajaj Dominar 400 ABS', 'Jawa Perak 330cc', 'KTM Duke 200cc', 'Bajaj Pulsar 180cc',
                   'TVS Apache RTR 200 4V', 'KTM Duke 390cc'],
    'Model_Year': [2017, 2016, 2012, 2017, 2020, 2012, 2016, 2020, 2018],
    'Kms_Driven': [17000, 50000, 14795, 28, 2000, 24561, 19718, 40, 1350],
    'Owner': ['first owner', 'first owner', 'first owner', 'first owner', 'first owner', 'third owner',
              'first owner', 'first owner', 'first owner'],
    'Location': ['hyderabad', 'hyderabad', 'hyderabad', 'pondicherry', 'bangalore', 'bangalore',
                 'bangalore', 'hyderabad', 'jaipur'],
    'Mileage': [35, 35, 30, 28, np.nan, 35, 65, 40, 25],
    'Power': [19, 19.8, 28, 34.5, 30, 25, 17, 20.21, 42.9],
    'Price': [63500, 115000, 300000, 100000, 197500, 63400, 55000, 120000, 198000]
}

df = pd.DataFrame(data)

# Step 2: Preprocess the data
# Convert categorical features to numerical
df['Owner'] = df['Owner'].map({'first owner': 1, 'second owner': 2, 'third owner': 3})
df['Location'] = df['Location'].map({'hyderabad': 1, 'pondicherry': 2, 'bangalore': 3, 'jaipur': 4})

# Fill missing values in Mileage with the mean value
df['Mileage'].fillna(df['Mileage'].mean(), inplace=True)

# Create target variable
df['Category'] = df['Price'].apply(lambda x: 'Expensive' if x > 100000 else 'Affordable')

# Features and target
X = df[['Model_Year', 'Kms_Driven', 'Owner', 'Location', 'Mileage', 'Power']]
y = df['Category']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 3: Classification without Dimensionality Reduction
# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

# Step 4: PCA + Classification
pca = PCA(n_components=2)
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.transform(X_test)

# Decision Tree with PCA
dt_pca_classifier = DecisionTreeClassifier(random_state=42)
dt_pca_classifier.fit(X_pca_train, y_train)
dt_pca_predictions = dt_pca_classifier.predict(X_pca_test)

# Random Forest with PCA
rf_pca_classifier = RandomForestClassifier(random_state=42)
rf_pca_classifier.fit(X_pca_train, y_train)
rf_pca_predictions = rf_pca_classifier.predict(X_pca_test)

# Step 5: LDA + Classification
lda = LDA(n_components=1)
X_lda_train = lda.fit_transform(X_train, y_train)
X_lda_test = lda.transform(X_test)

# Decision Tree with LDA
dt_lda_classifier = DecisionTreeClassifier(random_state=42)
dt_lda_classifier.fit(X_lda_train, y_train)
dt_lda_predictions = dt_lda_classifier.predict(X_lda_test)

# Random Forest with LDA
rf_lda_classifier = RandomForestClassifier(random_state=42)
rf_lda_classifier.fit(X_lda_train, y_train)
rf_lda_predictions = rf_lda_classifier.predict(X_lda_test)

# Step 6: Visualize the results
# Visualize the accuracy of each model
accuracies = {
    "Decision Tree": accuracy_score(y_test, dt_predictions),
    "Random Forest": accuracy_score(y_test, rf_predictions),
    "Decision Tree + PCA": accuracy_score(y_test, dt_pca_predictions),
    "Random Forest + PCA": accuracy_score(y_test, rf_pca_predictions),
    "Decision Tree + LDA": accuracy_score(y_test, dt_lda_predictions),
    "Random Forest + LDA": accuracy_score(y_test, rf_lda_predictions)
}

plt.figure(figsize=(10, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.title('Comparison of Classification Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()

# Print classification reports for detailed comparison
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_predictions))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
print("Decision Tree + PCA Classification Report:\n", classification_report(y_test, dt_pca_predictions))
print("Random Forest + PCA Classification Report:\n", classification_report(y_test, rf_pca_predictions))
print("Decision Tree + LDA Classification Report:\n", classification_report(y_test, dt_lda_predictions))
print("Random Forest + LDA Classification Report:\n", classification_report(y_test, rf_lda_predictions))
