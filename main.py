from data_loader import load_data
from model import SerieAModel
from difflib import get_close_matches

def find_team(name, teams):
    """Restituisce il nome della squadra più simile (case-insensitive)"""
    matches = get_close_matches(name.lower(), teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

def main():
    df = load_data(2018, 2022)
    model = SerieAModel()
    model.train(df)

    # Lista di tutte le squadre normalizzate
    all_teams = [t.lower() for t in model.le_team.classes_]

    print("\n⚠️ Squadre disponibili:")
    print(", ".join(sorted([t.title() for t in all_teams])))

    while True:
        print("\n⚽ Previsione partita Serie A")
        home = input("Inserisci la squadra di casa (oppure 'exit' per uscire): ")
        if home.lower() == "exit":
            print("👋 Ciao!")
            break
        away = input("Inserisci la squadra ospite: ")

        # Ricerca fuzzy per trovare la squadra più simile
        home_team_corrected = find_team(home, all_teams)
        away_team_corrected = find_team(away, all_teams)

        if home_team_corrected is None or away_team_corrected is None:
            print("❌ Squadra non presente nel dataset.")
        else:
            prediction = model.predict(home_team_corrected, away_team_corrected)
            print(f"🔮 Predizione: {home.title()} vs {away.title()} → {prediction}")

if __name__ == "__main__":
    main()