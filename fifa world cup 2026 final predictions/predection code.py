import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib, os, threading

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🏆 FIFA 2026 Winning Matches Prediction ML")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f8ff")
        self.frame = ttk.Frame(root, padding=20)
        self.frame.pack(fill="both", expand=True)

        self.df = None
        self.features_df = None
        self.lr = None
        self.rf = None
        self.scaler = None
        self.winner_label = None

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f8ff")
        style.configure("TLabel", background="#f0f8ff", foreground="#000080", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 10, "bold"), padding=8, background="#4682b4", foreground="white")
        style.map("TButton", background=[("active", "#5a9bd4")])

        self.create_widgets()

    def create_widgets(self):
        row = 0
        ttk.Label(self.frame, text="🏆 FIFA 2026 Winning Matches Prediction ML", font=("Arial", 24, "bold"), foreground="#000080").grid(column=0, row=row, columnspan=4, pady=20)
        row += 1
        ttk.Button(self.frame, text="Select and Load CSV File", command=lambda: threading.Thread(target=self.load_data).start()).grid(column=0, row=row, sticky="w", padx=10, pady=10)
        self.data_label = ttk.Label(self.frame, text="No file loaded")
        self.data_label.grid(column=1, row=row, columnspan=3, sticky="w")
        row += 1
        ttk.Button(self.frame, text="Generate Features", command=lambda: threading.Thread(target=self.preprocess).start()).grid(column=0, row=row, padx=10, pady=10, sticky="w")
        ttk.Button(self.frame, text="Train Models", command=lambda: threading.Thread(target=self.train_models).start()).grid(column=1, row=row, padx=10, pady=10, sticky="w")
        ttk.Button(self.frame, text="Evaluate Models", command=lambda: threading.Thread(target=self.evaluate).start()).grid(column=2, row=row, padx=10, pady=10, sticky="w")
        ttk.Button(self.frame, text="Predict 2026 Winner", command=lambda: threading.Thread(target=self.predict_latest).start()).grid(column=3, row=row, padx=10, pady=10, sticky="w")
        row += 1
        ttk.Button(self.frame, text="Open Output Folder", command=self.open_output_folder).grid(column=0, row=row, padx=10, pady=10, sticky="w")
        ttk.Button(self.frame, text="Exit", command=self.root.quit).grid(column=3, row=row, padx=10, pady=10, sticky="e")
        row += 1
        ttk.Label(self.frame, text="Log / Output", font=("Arial", 14, "bold")).grid(column=0, row=row, pady=10, sticky="w")
        row += 1
        self.log = tk.Text(self.frame, height=15, width=120, bg="#ffffff", fg="#000000", font=("Courier New", 10), relief="sunken", borderwidth=2)
        self.log.grid(column=0, row=row, columnspan=4)
        row += 1
        self.progress = ttk.Progressbar(self.frame, length=700, mode="determinate")
        self.progress.grid(column=0, row=row, columnspan=4, pady=10)
        row += 1
        self.winner_label = ttk.Label(self.frame, text="", font=("Arial", 20, "bold"), foreground="#ff4500", background="#f0f8ff")
        self.winner_label.grid(column=0, row=row, columnspan=4, pady=20)
        self.log_insert("Ready")

    def log_insert(self, t):
        self.log.insert("end", f"{t}\n")
        self.log.see("end")

    def load_data(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if not path:
                return
            self.df = pd.read_csv(path)
            self.data_label.config(text=os.path.basename(path))
            self.log_insert(f"Loaded {path} | Rows={self.df.shape[0]} Cols={self.df.shape[1]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def preprocess(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load dataset first")
            return
        self.progress['value'] = 20
        df = self.df.copy()
        df.columns = [col.strip() for col in df.columns]
        teams = pd.unique(df[['Home Team', 'Away Team']].values.ravel('K'))
        records = []
        for team in teams:
            matches_home = df[df['Home Team'] == team]
            matches_away = df[df['Away Team'] == team]
            total_matches = pd.concat([matches_home, matches_away])
            wins = sum((total_matches['Winner'] == team))
            losses = sum((total_matches['Loser'] == team))
            draws = sum((total_matches['Result'] == 'Draw'))
            goals_for = sum(total_matches.loc[total_matches['Home Team'] == team, 'Home Goals']) + \
                        sum(total_matches.loc[total_matches['Away Team'] == team, 'Away Goals'])
            goals_against = sum(total_matches.loc[total_matches['Home Team'] == team, 'Away Goals']) + \
                            sum(total_matches.loc[total_matches['Away Team'] == team, 'Home Goals'])
            reached_final = any((df[(df['Stage'].str.lower() == 'final')][['Home Team', 'Away Team']] == team).any(axis=1))
            won_final = any((df[(df['Stage'].str.lower() == 'final')]['Winner'] == team))
            records.append({
                'Team': team,
                'Wins': wins,
                'Losses': losses,
                'Draws': draws,
                'Goals_For': goals_for,
                'Goals_Against': goals_against,
                'Reached_Final': int(reached_final),
                'Won_Final': int(won_final)
            })
        features_df = pd.DataFrame(records)
        self.features_df = features_df
        features_df.to_csv("features_2026.csv", index=False)
        self.progress['value'] = 100
        self.log_insert("Preprocessing complete. Saved features_2026.csv")
        self.progress['value'] = 0

    def train_models(self):
        if self.features_df is None:
            messagebox.showwarning("No features", "Run preprocessing first")
            return
        df = self.features_df.copy()
        features = ['Wins', 'Draws', 'Losses', 'Goals_For', 'Goals_Against']
        X_train, y_train = df[features].values, df['Reached_Final'].values
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        self.lr = LogisticRegression(max_iter=500)
        self.rf = RandomForestClassifier(n_estimators=150, random_state=42)
        self.lr.fit(X_train_s, y_train)
        self.rf.fit(X_train, y_train)
        joblib.dump(self.lr, "lr_2026.joblib")
        joblib.dump(self.rf, "rf_2026.joblib")
        joblib.dump(self.scaler, "scaler_2026.joblib")
        self.log_insert("Training complete. Models for 2026 saved.")

    def evaluate(self):
        if self.features_df is None or self.lr is None or self.rf is None:
            messagebox.showwarning("Missing", "Run preprocessing and training first")
            return
        df = self.features_df.copy()
        features = ['Wins', 'Draws', 'Losses', 'Goals_For', 'Goals_Against']
        X, y = df[features].values, df['Reached_Final'].values
        Xs = self.scaler.transform(X)
        yp_lr = self.lr.predict(Xs)
        yp_rf = self.rf.predict(X)
        def score(yp, y):
            return {'acc': accuracy_score(y, yp), 'prec': precision_score(y, yp, zero_division=0),
                    'rec': recall_score(y, yp, zero_division=0), 'f1': f1_score(y, yp, zero_division=0)}
        s_lr = score(yp_lr, y)
        s_rf = score(yp_rf, y)
        self.log_insert(f"Logistic Regression (2026): acc={s_lr['acc']:.3f} prec={s_lr['prec']:.3f} rec={s_lr['rec']:.3f} f1={s_lr['f1']:.3f}")
        self.log_insert(f"Random Forest (2026): acc={s_rf['acc']:.3f} prec={s_rf['prec']:.3f} rec={s_rf['rec']:.3f} f1={s_rf['f1']:.3f}")
        self.log_insert("Evaluation complete for 2026 models.")

    def predict_latest(self):
        if self.features_df is None or self.rf is None:
            messagebox.showwarning("Missing", "Run preprocessing and training first")
            return
        df = self.features_df.copy()
        features = ['Wins', 'Draws', 'Losses', 'Goals_For', 'Goals_Against']
        X = df[features]
        probs = self.rf.predict_proba(X.values)[:, 1]
        df['Final_Prob'] = probs
        df = df.sort_values('Final_Prob', ascending=False)
        df.to_csv("predictions_rf_2026.csv", index=False)
        top5 = df.head(5)
        self.log_insert("Predicted Top 5 Teams Most Likely to Reach FIFA 2026 Final:")
        for _, row in top5.iterrows():
            self.log_insert(f"{row['Team']} ({row['Final_Prob']:.3f})")
        winner = top5.iloc[0]['Team']
        self.log_insert(f"🏆 Most Probable 2026 Winner: {winner}")
        self.winner_label.config(text=f"🏆 Predicted 2026 Winner: {winner}")
        self.log_insert("Predictions saved to predictions_rf_2026.csv")

    def open_output_folder(self):
        os.startfile(os.getcwd())

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
