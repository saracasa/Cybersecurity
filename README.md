# PROGETTO 7 - Privacy-preserving synthetic healthcare data generation

## Domanda di Ricerca

Può il dato sanitario sintetico generato per la protezione della sicurezza informatica mantenere una sufficienza fedeltà statistica e utilità di ricerca tali da sostituire i dati reali dei pazienti nella ricerca medica e nelle applicazioni di machine learning?

## Introduzione

Le violazioni dei dati sanitari sono sempre più comuni e devastanti, con cartelle cliniche vendute a oltre 250 dollari sui mercati del dark web rispetto ai 5 dollari delle carte di credito. Il GDPR e l'HIPAA impongono sanzioni severe per l'esposizione dei dati, eppure i ricercatori necessitano di accedere a dataset realistici per lo sviluppo dell'IA medica e per gli studi clinici.

La generazione di dati sintetici offre una soluzione di cybersecurity: se violati, i dataset sintetici non contengono alcuna informazione reale sui pazienti, minimizzando la responsabilità legale e il danno al paziente.

Tuttavia, il dato sintetico è prezioso solo se preserva le proprietà statistiche necessarie per una ricerca valida. Questo progetto valuta il compromesso fondamentale: è possibile generare dati sanitari sintetici che siano contemporaneamente sicuri contro gli attacchi alla privacy (re-identificazione, inferenza di appartenenza) e utili per la ricerca legittima (mantengono correlazioni, addestrano modelli di ML accurati, consentono un'analisi statistica valida)?

Lo studio confronterà diversi metodi di generazione—dalle semplici approcci statistici a GAN avanzate con privacy differenziale—misurando sia le loro proprietà di sicurezza (resistenza agli attacchi) sia la loro utilità di ricerca (somiglianza statistica, prestazioni di ML). I risultati determineranno quali tecniche bilanciano meglio la protezione della privacy con l'utilità del dato.

## Guida all'implementazione

Quadro di confronto principale:

- Generare dataset sintetici usando metodi con diversi livelli di privacy (nessuna privacy, moderata, forte)
- Verificare la somiglianza statistica: confrontare distribuzioni, matrici di correlazione e test statistici standard
- Valutare l'utilità per la ricerca: addestrare modelli di predizione delle malattie su dati sintetici e valutarli su un set reale di holdout
- Valutare la privacy: implementare attacchi di membership inference per misurare la perdita di informazioni
- Visualizzare il trade-off: tracciare livello di privacy vs. metriche di utilità per identificare il bilanciamento ottimale

## Dataset pubblici

1. UCI Diabetes Dataset — RACCOMANDATO
    - Link: <https://archive.ics.uci.edu/dataset/34/diabetes>
    - Descrizione: Dataset piccolo e facilmente gestibile, ideale per prototipare metodi.
    - Contiene: 768 pazienti con predittori clinici e outcome sul diabete.
    - Ideale per: Sviluppo e test iniziali dei metodi.

2. UCI Heart Disease Dataset
    - Link: <https://archive.ics.uci.edu/dataset/45/heart+disease>
    - Descrizione: Dati multi‑attributo sulla salute cardiovascolare.
    - Contiene: Caratteristiche cliniche e diagnosi di cardiopatia.
    - Ideale per: Verificare la generalizzazione su domini medici diversi.

3. MIMIC‑III Clinical Database (Avanzato)
    - Link: <https://physionet.org/content/mimiciii/>
    - Descrizione: Cartelle cliniche reali di terapia intensiva provenienti da oltre 40.000 pazienti.
    - Contiene: Demografia, segni vitali, esami di laboratorio, farmaci, diagnosi.
    - Nota: L'accesso richiede accreditamento e approvazione etica per uso di ricerca.
    - Ideale per: Valutazioni complete su dati complessi e realistici.

## Tool e risorse suggerite

- Librerie per la generazione di dati sintetici: SDV (Synthetic Data Vault) con CTGAN; synthcity (con opzioni per la privacy)
- Strumenti per la privacy: libreria Google Differential Privacy; diffprivlib (IBM)
- Valutazione: scikit-learn per test ML; scipy.stats per confronti statistici

## Referenze chiave

- Stadler, T., et al. (2022). "Synthetic Data - Anonymisation Groundhog Day."
USENIX Security. [Privacy vulnerabilities in synthetic data]
- Jordon, J., et al. (2022). "Synthetic Data - What, Why and How?" The Royal
Society. [Comprehensive overview with healthcare focus]
- Xu, L., et al. (2019). "Modeling Tabular Data using Conditional GAN."
NeurIPS. [CTGAN baseline method]
- Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential
Privacy." Foundations and Trends in TCS. [Privacy theory fundamentals]