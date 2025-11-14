import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import numpy as np
import seaborn as sns

# Carica il dataset
df = pd.read_csv("dataset/raw/diabetes.csv")
print("Dataset caricato. Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.dtypes) 

# Sostituisci 0 con NaN e imputazione con mediana
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

# Crea e rileva i metadati
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Forza colonne categoriali o numeriche per sicurezza
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

# Salva i dati sintetici
synthetic.to_csv("dataset/synthetic/diabetes_synth_noprivacy.csv", index=False)
print("Dati sintetici generati e salvati.")
