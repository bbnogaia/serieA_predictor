from data_loader import load_data
from model import SerieAModel

def add_result_column(df):
    """
    Aggiunge la colonna 'result' al DataFrame:
    - 'H' se vittoria casa
    - 'A' se vittoria ospite
    - 'D' se pareggio
    """
    def compute_result(row):
        if row['home_goals'] > row['away_goals']:
            return 'H'
        elif row['home_goals'] < row['away_goals']:
            return 'A'
        else:
            return 'D'

    df['result'] = df.apply(compute_result, axis=1)
    return df


def main():
    df = load_data(2018, 2022)
    df = add_result_column(df)
    model = SerieAModel()
    model.train(df)

    while True:
        print("\n⚽ Previsione partita Serie A")
        home = input("Inserisci la squadra di casa (oppure 'exit' per uscire): ")
        if home.lower() == "exit":
            print("👋 Ciao!")
            break
        away = input("Inserisci la squadra ospite: ")

        prediction = model.predict(home, away)
        print(f"🔮 Predizione: {home} vs {away} → {prediction}")

if __name__ == "__main__":
    main()