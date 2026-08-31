import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score
import os
# os.chdir("...")  # Optionnel : changer le répertoire de travail


# ==========================================================
# 1. LECTURE DU CSV ET PRÉPARATION DES DONNÉES
# ==========================================================

# Lecture du fichier : "Date" est convertie en datetime automatiquement
df = pd.read_csv("sp500_20y_daily_after_ACP.csv", parse_dates=["Date"])

# Sélection des features :
# On prend toutes les colonnes sauf Date et Price
features = [col for col in df.columns if col not in ["Date", "Price"]]

# Horizon de prédiction : rendement à 21 jours
nb_days_target = 21

# Calcul du rendement 21 jours plus tard : (P(t+21) / P(t)) - 1
df["21_day_return"] = df["Price"].shift(-nb_days_target) / df["Price"] - 1

# On enlève les lignes pour lesquelles on n’a pas de rendement futur
df = df.dropna(subset=["21_day_return"])

# Construction de la cible binaire :
# 1 = rendement positif dans 21 jours
# 0 = rendement négatif
df["direction_rendement"] = (df["21_day_return"] > 0).astype(int)
target = "direction_rendement"


# ==========================================================
# 2. PARAMÈTRES POUR LE WALK-FORWARD
# ==========================================================

nb_dates = len(df["Date"])        # Nombre total d'observations
test_start_date = 250             # 1ère prédiction après 250 jours

# Listes où on stockera les résultats du walk-forward
predictions = []
vraies_valeurs = []
dates_test = []
training_accuracy = []
train_direction_accuracy = []
feature_importances_list = []     # Importance des features à chaque fenêtre


# ==========================================================
# 3. BOUCLE WALK-FORWARD (modèle ré-entraîné à chaque jour)
# ==========================================================

for i in range(test_start_date, nb_dates - nb_days_target):

    # Définit les dates de début et fin de l'échantillon d'entraînement
    start_date_train = df["Date"].iloc[i - test_start_date]
    end_date_train   = df["Date"].iloc[i - 1]

    # Date du point à prédire
    test_date = df["Date"].iloc[i]

    # Extraction de la fenêtre glissante d'entraînement
    train = df[(df["Date"] >= start_date_train) & (df["Date"] <= end_date_train)]

    # Extraction de la ligne unique servant de test
    test = df[df["Date"] == test_date]

    # On retire les NA éventuels dans la target
    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])

    # Séparation X / y
    X_train = train[features]
    y_train = train[target]
    X_test  = test[features]
    y_test  = test[target]

    # ------------------------------------------------------
    # 3.1 Définition du XGBoost (paramètres anti-overfitting)
    # ------------------------------------------------------

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        min_child_weight=5,
        reg_lambda=5,
        reg_alpha=1,
        colsample_bytree=0.7,
        objective="binary:logistic",
        random_state=42,
    )

    # Entraînement
    model.fit(X_train, y_train)

    # Sauvegarde de l’importance des features du modèle courant
    feature_importances_list.append(model.feature_importances_)

    # ------------------------------------------------------
    # 3.2 Accuracy sur l’échantillon d’entraînement
    # ------------------------------------------------------
    y_train_predicted = model.predict(X_train)
    train_dir_acc = (y_train_predicted == y_train).mean()
    train_direction_accuracy.append(train_dir_acc)

    # ------------------------------------------------------
    # 3.3 Prédiction du jour courant (une seule observation)
    # ------------------------------------------------------
    y_predicted = model.predict(X_test)[0]

    predictions.append(y_predicted)
    vraies_valeurs.append(y_test.iloc[0])
    dates_test.append(test_date)


# ==========================================================
# 4. AGRÉGATION DES RÉSULTATS
# ==========================================================

array_predictions = np.array(predictions)
array_true_values = np.array(vraies_valeurs)

# Accuracy moyenne sur l'entraînement
train_direction_acc_mean = np.mean(train_direction_accuracy)

# Accuracy du walk-forward
direction_acc = (array_predictions == array_true_values).mean()

print("Train direction accuracy :", train_direction_acc_mean)
print("Validation direction accuracy :", direction_acc)


# ==========================================================
# 5. ANALYSE DES RENDEMENTS NÉGATIFS
# ==========================================================

true_neg = array_true_values == 0     # Vrai rendement négatif
pred_neg = array_predictions == 0     # Modèle prédit négatif

# Recall négatif : parmi les vrais baisses, combien bien prédits ?
recall_neg = (pred_neg[true_neg]).mean() if true_neg.sum() > 0 else np.nan

# Précision négative : parmi les prédictions de baisse, combien vraies ?
precision_neg = (array_true_values[pred_neg] == 0).mean() if pred_neg.sum() > 0 else np.nan


# ==========================================================
# 6. RATIO DE JOURS HAUSSIERS (baseline buy & hold)
# ==========================================================

ratio = (array_true_values == 1).mean()


print("Recall négatif :", recall_neg)
print("Précision négative :", precision_neg)
print("Ratio haussier :", ratio)


# ==========================================================
# 7. GRAPHIQUE : Matrice de confusion
# ==========================================================

cm = confusion_matrix(array_true_values, array_predictions)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Prédit baisse", "Prédit hausse"],
            yticklabels=["Vrai baisse", "Vrai hausse"])
plt.title("Matrice de confusion")
plt.show()


# ==========================================================
# 8. IMPORTANCE MOYENNE DES FEATURES
# ==========================================================

feature_importances_array = np.vstack(feature_importances_list)
mean_feature_importance = feature_importances_array.mean(axis=0)

plt.figure(figsize=(10, 5))
plt.bar(range(len(features)), mean_feature_importance)
plt.xticks(range(len(features)), features, rotation=90)
plt.ylabel("Importance moyenne")
plt.title("Importance moyenne des features (XGBoost, toutes fenêtres)")
plt.tight_layout()
plt.show()


# ==========================================================
# 9. BACKTEST : BUY & HOLD vs STRATÉGIE BASÉE SUR LE MODÈLE
# ==========================================================

results_df = pd.DataFrame({
    "Date": dates_test,
    "Pred_direction": array_predictions
})

# Ajoute le vrai rendement futur à ces dates
df_returns = df[["Date", "21_day_return"]]
results_df = results_df.merge(df_returns, on="Date", how="left")
results_df = results_df.sort_values("Date").reset_index(drop=True)

initial_capital = 10000
dates_strategy = []
bh_values = []
model_values = []

portfolio_bh = initial_capital
portfolio_model = initial_capital

# On avance par pas de 21 jours (réévaluation mensuelle)
for i in range(0, len(results_df), nb_days_target):
    row = results_df.iloc[i]
    true_return = row["21_day_return"]
    pred = row["Pred_direction"]
    date = row["Date"]

    # Buy & Hold : toujours investi
    portfolio_bh *= (1 + true_return)

    # Modèle : investi seulement si prédiction haussière
    if pred == 1:
        portfolio_model *= (1 + true_return)

    dates_strategy.append(date)
    bh_values.append(portfolio_bh)
    model_values.append(portfolio_model)

# ----------------------------------------------------------
# 9.1 Graphique des performances cumulées
# ----------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(dates_strategy, bh_values, label="Buy & Hold SP500 (10 000 $)")
plt.plot(dates_strategy, model_values, label="Stratégie modèle 21j (10 000 $)", linestyle="--")

plt.xlabel("Date")
plt.ylabel("Valeur du portefeuille ($)")
plt.title("Buy & Hold vs Stratégie basée sur le modèle (réévaluation 21 jours)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Valeur finale Buy & Hold :", bh_values[-1])
print("Valeur finale Stratégie modèle :", model_values[-1])
