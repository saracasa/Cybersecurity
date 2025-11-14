import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import numpy as np
import seaborn as sns

# Carica il dataset
df = pd.read_csv("dataset/raw/diabetes.csv")
print("Dataset caricato. Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.dtypes) # Controlla tipi

# Sostituisci 0 con NaN e imputazione con mediana
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

#df['Insulin'] = np.log1p(df['Insulin'])
#df['Glucose'] = np.log1p(df['Glucose'])

# Crea e rileva i metadati
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Forza colonne categoriali o numeriche
metadata.update_column(
    column_name='Outcome',
    sdtype='categorical'
)

for col in ['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']:
    metadata.update_column(column_name=col, sdtype='numerical')

# Crea e addestra il modello CTGAN
model = CTGANSynthesizer(
    metadata,
    epochs=2000,                 
    batch_size=500,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256),
    verbose=True
)
model.fit(df)

# Genera dati sintetici
synthetic= model.sample(num_rows=len(df))
print("Synthetic shape:", synthetic.shape)
fig = model.get_loss_values_plot()
fig.show()

# Salva i dati sintetici
synthetic.to_csv("dataset/synthetic/diabetes_synth_noprivacy.csv", index=False)
print("Dati sintetici generati e salvati.")

from sdmetrics.reports.single_table import QualityReport

# Crea e genera il report di qualità
report = QualityReport()
report.generate(real_data=df, synthetic_data=synthetic, metadata=metadata.to_dict())

# Mostra il punteggio complessivo (da 0 a 1)
print("\nQualità complessiva:", round(report.get_score(), 3))

# Dettagli per ogni colonna (forma e relazione)
details = report.get_details(property_name='Column Shapes')
print("\nQualità per colonna (forme):")
print(details[['Column', 'Score']])

import matplotlib.pyplot as plt

cols_to_plot = ['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(9, 4*len(cols_to_plot)))

for i, col in enumerate(cols_to_plot):
    ax = axes[i]
    sns.histplot(df[col], color='blue', label='Reale', kde=True, stat="density", bins=30, alpha=0.5, ax=ax)
    sns.histplot(synthetic[col], color='orange', label='Sintetico', kde=True, stat="density", bins=30, alpha=0.5, ax=ax)
    ax.set_title(f'Distribuzione: {col}')
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig('compare_distributions_all_features.png', dpi=150)
plt.close(fig)

#aggiungi report 