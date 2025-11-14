import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import numpy as np
from sdmetrics.reports.single_table import QualityReport
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality  
from sdv.metadata import SingleTableMetadata

# Carica i dati
real = pd.read_csv('dataset/raw/diabetes.csv')
synth = pd.read_csv('dataset/synthetic/diabetes_synth_noprivacy.csv')

# Sostituisci 0 con NaN e poi con mediana per colonne specifiche
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
real[cols_with_zeros] = real[cols_with_zeros].replace(0, np.nan)
real[cols_with_zeros] = real[cols_with_zeros].fillna(real[cols_with_zeros].median())

# Colonne numeriche da analizzare
num_cols = [col for col in real.columns if pd.api.types.is_numeric_dtype(real[col])]

# Crea e rileva i metadati
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real)

# Forza colonne categoriali o numeriche
metadata.update_column(
    column_name='Outcome',
    sdtype='categorical'
)

for col in ['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']:
    metadata.update_column(column_name=col, sdtype='numerical')


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

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Prima riga: 
sns.heatmap(real_corr, ax=axes[0, 0], cmap='coolwarm', annot=True, fmt='.2f', annot_kws={"size":8})
axes[0, 0].set_title('Correlazione Reale', fontsize=10)
axes[0, 0].tick_params(axis='x', labelsize=5, rotation=60)
axes[0, 0].tick_params(axis='y', labelsize=5)
sns.heatmap(synth_corr, ax=axes[0, 1], cmap='coolwarm', annot=True, fmt='.2f', annot_kws={"size":8})
axes[0, 1].set_title('Correlazione Sintetico', fontsize=10)
axes[0, 1].tick_params(axis='x', labelsize=5, rotation=60)
axes[0, 1].tick_params(axis='y', labelsize=5)

# Seconda riga:
axes[1, 1].axis('off')
sns.heatmap(abs(real_corr - synth_corr), ax=axes[1, 0], cmap='viridis', annot=True, fmt='.2f', annot_kws={"size":8})
axes[1, 0].set_title('Differenza Assoluta Correlazioni', fontsize=10)
axes[1, 0].tick_params(axis='x', labelsize=5, rotation=60)
axes[1, 0].tick_params(axis='y', labelsize=5)

plt.tight_layout(pad=5)
plt.show()


# 4. Confronto MEDIA per colonna
print("\nConfronto delle medie tra reale e sintetico:")
print("{:<28} {:>12} {:>16} {:>12}".format('Feature', 'Mean Real', 'Mean Synthetic', 'Diff'))
for col in num_cols:
    mean_real = real[col].mean()
    mean_synth = synth[col].mean()
    diff = mean_synth - mean_real
    print("{:<28} {:>12.4f} {:>16.4f} {:>12.4f}".format(col, mean_real, mean_synth, diff))

# 5. REPORT QUALITA' dei dati sintetici
report = QualityReport()
report.generate(real_data=real, synthetic_data=synth, metadata=metadata.to_dict())

print("\nQualità complessiva:", round(report.get_score(), 3))

details = report.get_details(property_name='Column Shapes')
print("\nQualità per colonna (forme):")
print(details[['Column', 'Score']])

#Restituisce punteggi e report che indicano se i dati sintetici rispettano i vincoli, i formati e le regole definite dai metadati
diagnostic = run_diagnostic(
    real_data=real,
    synthetic_data=synth,
    metadata=metadata
)

#Misura la somiglianza statistica tra dati reali e sintetici
quality_report = evaluate_quality(
    real,
    synth,
    metadata
)

