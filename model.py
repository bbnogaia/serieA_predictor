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
        """Aggiunge la colonna 'result': H = vittoria casa, A = vittoria ospite, D = pareggio"""
        def compute_result(row):
            if row['home_goals'] > row['away_goals']:
                return 'H'
            elif row['home_goals'] < row['away_goals']:
                return 'A'
            else:
                return 'D'

        df['result'] = df.apply(compute_result, axis=1)
        return df

    def normalize_teams(self, df):
        df['home_team'] = df['home_team'].str.strip().str.lower()
        df['away_team'] = df['away_team'].str.strip().str.lower()
        return df

    def train(self, df):
        print("🎯 Addestramento del modello...")
        df = self.normalize_teams(df)
        df = self.add_result_column(df)

        all_teams = pd.concat([df['home_team'], df['away_team']])
        self.le_team.fit(all_teams)
        df['home_team_enc'] = self.le_team.transform(df['home_team'])
        df['away_team_enc'] = self.le_team.transform(df['away_team'])

        df['result_enc'] = self.le_result.fit_transform(df['result'])

        X = df[['home_team_enc', 'away_team_enc']]
        y = df['result_enc']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy modello: {acc:.2f}")

    def predict(self, home_team, away_team):
        home_team = home_team.strip().lower()
        away_team = away_team.strip().lower()

        if home_team not in self.le_team.classes_ or away_team not in self.le_team.classes_:
            return "❌ Squadra non presente nel dataset."

        home_enc = self.le_team.transform([home_team])[0]
        away_enc = self.le_team.transform([away_team])[0]

        pred = self.model.predict([[home_enc, away_enc]])[0]
        result = self.le_result.inverse_transform([pred])[0]
        return result

    def predict_proba(self, home_team, away_team):
        """Restituisce le probabilità di vittoria casa, pareggio e vittoria ospite"""
        home_team = home_team.strip().lower()
        away_team = away_team.strip().lower()

        if home_team not in self.le_team.classes_ or away_team not in self.le_team.classes_:
            return None

        home_enc = self.le_team.transform([home_team])[0]
        away_enc = self.le_team.transform([away_team])[0]

        probs = self.model.predict_proba([[home_enc, away_enc]])[0]
        # Associa le probabilità ai risultati corretti
        result_labels = self.le_result.inverse_transform(range(len(probs)))
        return dict(zip(result_labels, probs))