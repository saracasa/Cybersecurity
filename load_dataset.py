import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset Pima Indians Diabetes CSV
df = pd.read_csv('dataset/raw/diabetes.csv')

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

