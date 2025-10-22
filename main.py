from data_loader import load_data
from model import SerieAModel
from difflib import get_close_matches

def find_team(name, teams):
    matches = get_close_matches(name.lower(), teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

def main():
    df = load_data(2018, 2022)
    model = SerieAModel()
    model.train(df)

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

        home_team_corrected = find_team(home, all_teams)
        away_team_corrected = find_team(away, all_teams)

        if home_team_corrected is None or away_team_corrected is None:
            print("❌ Squadra non presente nel dataset.")
            continue

        result = model.predict(home_team_corrected, away_team_corrected)
        probs = model.predict_proba(home_team_corrected, away_team_corrected)

        print(f"🔮 Predizione: {home.title()} vs {away.title()} → {result}")
        if probs:
            print("📊 Probabilità:")
            for k, v in probs.items():
                print(f"  {k}: {v:.2f}")

if __name__ == "__main__":
    main()