import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Data
data = {
    'Model Name': [
        'Bajaj Avenger Cruise 220', 'Royal Enfield Classic 350cc', 'Hyosung GT250R', 'Bajaj Dominar 400 ABS', 
        'Jawa Perak 330cc', 'KTM Duke 200cc', 'Bajaj Pulsar 180cc', 'TVS Apache RTR 200 4V Dual Channel ABS BS6', 
        'KTM Duke 390cc'
    ],
    'Model Year': [2017, 2016, 2012, 2017, 2020, 2012, 2016, 2020, 2018],
    'Kms Driven': [17000, 50000, 14795, 28, 2000, 24561, 19718, 40, 1350],
    'Owner': ['First owner', 'First owner', 'First owner', 'First owner', 'First owner', 'Third owner', 'First owner', 'First owner', 'First owner'],
    'Location': ['Hyderabad', 'Hyderabad', 'Hyderabad', 'Pondicherry', 'Bangalore', 'Bangalore', 'Bangalore', 'Hyderabad', 'Jaipur'],
    'Mileage': ['35 kmpl', '35 kmpl', '30 kmpl', '28 Kms', '', '35 kmpl', '65 kmpl', '40 Kmpl', '25 kmpl'],
    'Power': ['19 bhp', '19.80 bhp', '28 bhp', '34.50 bhp', '30 bhp', '25 bhp', '17 bhp', '20.21 bhp', '42.90 bhp'],
    'Price': [63500, 115000, 300000, 100000, 197500, 63400, 55000, 120000, 198000]
}

df = pd.DataFrame(data)

# Preprocessing
# Convert categorical columns to numerical
le = LabelEncoder()
df['Owner'] = le.fit_transform(df['Owner'])
df['Location'] = le.fit_transform(df['Location'])

# Convert 'Mileage' and 'Power' to numeric values
df['Mileage'] = df['Mileage'].str.extract(r'(\d+)', expand=False).astype(float).fillna(0)
df['Power'] = df['Power'].str.extract(r'(\d+.\d+)', expand=False).astype(float).fillna(0)

# Prepare features
features = df[['Model Year', 'Kms Driven', 'Owner', 'Location', 'Mileage', 'Power', 'Price']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Apply LDA
n_classes = len(df['Owner'].unique())
lda_components = min(n_classes - 1, scaled_features.shape[1])  # LDA components should be min(n_classes - 1, n_features)
lda = LinearDiscriminantAnalysis(n_components=lda_components)
lda_result = lda.fit_transform(scaled_features, df['Owner'])

# Prepare Data for Plotting
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_lda = pd.DataFrame(lda_result, columns=[f'LD{i+1}' for i in range(lda_components)])
df_pca['Target'] = df['Owner']
df_lda['Target'] = df['Owner']

# Visualization
plt.figure(figsize=(12, 6))

# PCA Plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Target'], cmap='viridis', edgecolor='k', s=50)
plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Owner')

# LDA Plot
plt.subplot(1, 2, 2)
scatter = plt.scatter(df_lda.iloc[:, 0], df_lda.iloc[:, 1], c=df_lda['Target'], cmap='viridis', edgecolor='k', s=50)
plt.title('LDA')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.colorbar(scatter, label='Owner')

plt.tight_layout()
plt.show()
