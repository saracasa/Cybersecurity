import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

# Carica il dataset Pima Indians Diabetes CSV
df = pd.read_csv('dataset/real/diabetes.csv')

# Sostituisci 0 con NaN e imputazione con mediana
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

# Esplorazione base
print("Dimensioni dataset:", df.shape)
print("Info dataset:")
print(df.info())
print("Prime righe:")
print(df.head())

# Statistiche descrittive
print("Statistica descrittiva:")
print(df.describe())

# Distribuzione delle variabili numeriche
df.hist(figsize=(12, 10), bins=30)
plt.suptitle('Distribuzioni variabili')
plt.tight_layout()
plt.show()

# Matrice di correlazione
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matrice di correlazione')
plt.show()

# Suddivide il dataset in train (70%) e holdout (30%)
train_df, holdout_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Outcome'])

# Salva i due dataset in CSV separati
train_df.to_csv('dataset/real/diabetes_train.csv', index=False)
holdout_df.to_csv('dataset/real/diabetes_holdout.csv', index=False)

print(f"Dimensione train: {train_df.shape}")
print(f"Dimensione holdout: {holdout_df.shape}")

