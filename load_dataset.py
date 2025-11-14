import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset Pima Indians Diabetes CSV
df = pd.read_csv('dataset/raw/diabetes.csv')

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

