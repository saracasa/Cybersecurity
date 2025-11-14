import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import numpy as np

# Carica i dati
real = pd.read_csv('dataset/raw/diabetes.csv')
synth = pd.read_csv('dataset/synthetic/diabetes_synth_noprivacy.csv')

# Sostituisci 0 con NaN e poi con mediana
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
real[cols_with_zeros] = real[cols_with_zeros].replace(0, np.nan)
real[cols_with_zeros] = real[cols_with_zeros].fillna(real[cols_with_zeros].median())

#real['Insulin'] = np.log1p(real['Insulin'])
#real['Glucose'] = np.log1p(real['Glucose'])


num_cols = [col for col in real.columns if pd.api.types.is_numeric_dtype(real[col])]

# 1. Confronto DISTRIBUZIONI
fig, axes = plt.subplots(len(num_cols), 1, figsize=(9, 4*len(num_cols)))
for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.histplot(real[col], color='blue', label='Reale', kde=True, stat="density", bins=30, alpha=0.5, ax=ax)
    sns.histplot(synth[col], color='orange', label='Sintetico', kde=True, stat="density", bins=30, alpha=0.5, ax=ax)
    ax.set_title(f'Distribuzione: {col}')
    ax.set_ylabel('Density')
    ax.set_xlabel(col)
    ax.legend()
plt.tight_layout()
plt.savefig('compare_distributions.png', dpi=150)
plt.close(fig)

# 2. Confronto MATRICE DI CORRELAZIONE
real_corr = real[num_cols].corr()
synth_corr = synth[num_cols].corr()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', annot=True)
axes[0].set_title('Correlazione Reale')
sns.heatmap(synth_corr, ax=axes[1], cmap='coolwarm', annot=True)
axes[1].set_title('Correlazione Sintetico')
sns.heatmap(abs(real_corr - synth_corr), ax=axes[2], cmap='viridis', annot=True)
axes[2].set_title('Differenza Assoluta Correlazioni')
plt.tight_layout()
plt.show()

# 2. Confronto media per colonna
print("\nConfronto delle medie tra reale e sintetico:")
print("{:<28} {:>12} {:>16} {:>12}".format(
    'Feature', 'Mean Real', 'Mean Synthetic', 'Diff'))
for col in num_cols:
    mean_real = real[col].mean()
    mean_synth = synth[col].mean()
    diff = mean_synth - mean_real
    print("{:<28} {:>12.4f} {:>16.4f} {:>12.4f}".format(
        col, mean_real, mean_synth, diff))
