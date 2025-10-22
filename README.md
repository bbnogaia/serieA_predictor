# Serie A Match Predictor

[🇮🇹 Italiano](#italiano) | [🇬🇧 English](#english)

## Italiano

Questo progetto implementa un modello di **Machine Learning** per predire gli esiti delle partite del campionato di calcio italiano di **Serie A**.

Il modello utilizza dati storici delle stagioni **2018-2022** per classificare il risultato finale in tre categorie: **Vittoria Casa (H)**, **Pareggio (D)** o **Vittoria Ospite (A)**.

---

## Installazione e Utilizzo

Per configurare l'ambiente e poter iniziare ad usare il predittore:

1.  **Clona il repository:**

    ```bash
    git clone <url-del-tuo-repo>
    cd football-prediction-serieA
    ```

2.  **Crea e attiva un ambiente virtuale (consigliato):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # per Mac/Linux
    venv\Scripts\activate     # per Windows
    ```

3.  **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```
    _Dipendenze principali: `pandas`, `scikit-learn`, `kagglehub`, `difflib`._

### Come Usare

Per eseguire lo script da terminale:

```bash
python3 src/main.py
```

Il programma cerca di guidare l'utente nella scelta delle squadre, utilizzando una ricerca fuzzy integrata per gestire nomi inseriti in minuscolo o con leggere differenze ortografiche.
Un esempio di output è il seguente:

```bash
Previsione partita Serie A
Inserisci la squadra di casa (oppure 'exit' per uscire): fiorentina
Inserisci la squadra ospite: milan
Predizione: Fiorentina vs Milan → H
Probabilità:
  H: 0.45  (Vittoria casa)
  D: 0.30  (Pareggio)
  A: 0.25  (Vittoria ospite)
```

**Note sull'Accuracy**: L'accuratezza del modello è basata su un set di dati di training del 2018-2022 quindi è di circa 0.44. Il modello ha uno scopo principalmente didattico e non è ottimizzato per previsioni ad alta precisione.

## English

This project implements a **Machine Learning** model to predict the outcomes of Italian **Serie A** football matches.

The model uses historical data from the **2018-2022** seasons to classify the final result into three categories: **Home Win (H)**, **Draw (D)**, or **Away Win (A)**.

---

## Installation and Usage

Follow these steps to set up the environment and start using the predictor:

1.  **Clone the repository:**

    ```bash
    git clone <url-del-tuo-repo>
    cd football-prediction-serieA
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # for Mac/Linux
    venv\Scripts\activate     # for Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Key dependencies: `pandas`, `scikit-learn`, `kagglehub`, `difflib`._

### How to Use

To run the script from your terminal:

```bash
python3 src/main.py
```

The program attempts to guide the user in selecting the teams, using integrated fuzzy search to handle team names entered in lowercase or with slight spelling variations.
An example of the output is as follows:

```bash
Previsione partita Serie A
Inserisci la squadra di casa (oppure 'exit' per uscire): fiorentina
Inserisci la squadra ospite: milan
Predizione: Fiorentina vs Milan → H
Probabilità:
H: 0.45  (Vittoria casa)
D: 0.30  (Pareggio)
A: 0.25  (Vittoria ospite)
```

**Accuracy Note**: The model's accuracy is based on a 2018-2022 training dataset and is approximately 0.44. The model is primarily for educational purposes and is not optimized for high-precision forecasts.
