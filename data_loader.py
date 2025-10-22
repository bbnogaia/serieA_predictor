import kagglehub
import pandas as pd
import os

def load_data(start_season=2018, end_season=2022):
    print("📥 Caricamento dataset Serie A...")

    # Scarica il dataset da Kaggle
    path = kagglehub.dataset_download("giovannicarlozzi/serie-a-matches-dataset")

    dfs = []

    # Esplora tutti i file CSV nella cartella scaricata
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            # Estrai l'anno dal nome del file
            try:
                season = int(filename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue  # Ignora file non conformi

            if start_season <= season <= end_season:
                # Leggi il CSV con separatore corretto
                df = pd.read_csv(os.path.join(path, filename), sep=';')
                # Aggiungi la colonna season
                df['season'] = season
                dfs.append(df)

    if not dfs:
        raise FileNotFoundError("❌ Nessun file CSV trovato per le stagioni richieste.")

    # Combina tutti i DataFrame in uno solo
    df = pd.concat(dfs, ignore_index=True)

    # Mantieni solo le colonne essenziali e rimuovi righe incomplete
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    print(f"✅ Dati caricati: {len(df)} partite dal {start_season} al {end_season}")

    return df