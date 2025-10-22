from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

class SerieAModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.le_team = LabelEncoder()
        self.le_result = LabelEncoder()

    def add_result_column(self, df):
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

    def train(self, df):
        print("🎯 Addestramento del modello...")

        # Aggiungi la colonna result
        df = self.add_result_column(df)

        # Codifica tutte le squadre insieme
        all_teams = pd.concat([df['home_team'], df['away_team']])
        self.le_team.fit(all_teams)
        df['home_team_enc'] = self.le_team.transform(df['home_team'])
        df['away_team_enc'] = self.le_team.transform(df['away_team'])

        # Codifica il target
        df['result_enc'] = self.le_result.fit_transform(df['result'])

        # Features e target
        X = df[['home_team_enc', 'away_team_enc']]
        y = df['result_enc']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Addestra il modello
        self.model.fit(X_train, y_train)

        # Valutazione
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy modello: {acc:.2f}")

    def predict(self, home_team, away_team):
        # Controlla che le squadre siano presenti
        if home_team not in self.le_team.classes_ or away_team not in self.le_team.classes_:
            return "❌ Squadra non presente nel dataset."

        home_enc = self.le_team.transform([home_team])[0]
        away_enc = self.le_team.transform([away_team])[0]
        pred = self.model.predict([[home_enc, away_enc]])[0]
        result = self.le_result.inverse_transform([pred])[0]
        return result