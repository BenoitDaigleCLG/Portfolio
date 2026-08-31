import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def affiche_correlations(data, title):
    """
    Affiche une heatmap des corrélations entre variables numériques.

    """
    colonnes_numeriques = data.select_dtypes(include=[np.number])
    matrice_corr = colonnes_numeriques.corr()
    # Pour voir juste sous la diagonale (matrice symétrique)
    masque = np.triu(np.ones_like(matrice_corr, dtype=bool))

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        matrice_corr,
        mask=masque,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


df = pd.read_excel("../Data/randomforest.xlsx")

# On enlève les variables qu'on ne veut pas utiliser dans le modèle
cols_to_drop = ["PLAYER_ID", "DRAFT_OVERALL", "DRAFT_YEAR"]

X = df.drop(columns=cols_to_drop)
y = df["DRAFT_OVERALL"]

# Pour voir les corrélations entre les variables
affiche_correlations(X, "Détection de le colinéarité avant traitement")

# Colonnes qu'on décide d'enlever suite à la colinéarité
colonnes_supp_to_drop = [
    "PPG",
    "GPG",
    "MAX_PPG",
    "LAST_G_PER_GP",
    "LAST_A_PER_GP",
    "APG",
    "PIM_PGP",
    "BMI_PROXY",
    "SNIPER_RATIO",
]

X = X.drop(columns=colonnes_supp_to_drop)
# On affiche pour voir qu'on a bien retiré certaines variables qui souffraient de colinéarité
affiche_correlations(X, "Détection de le colinéarité après traitement")

# Les variables devant être converties en dummies
X = pd.get_dummies(X, columns=["PRIMARY_POS", "SHOOTS", "LAST_LEAGUE", "NATIONALITY"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Nombre de valeurs d'entrainement: ", len(X_train))
print("Nombre de valeurs de test: ", len(X_test))

grille_parametres = {
    "n_estimators": [300, 500, 700, 900],
    "max_depth": [5, 10, 15, None],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2", 0.4, 0.7],
    "criterion": ["squared_error", "absolute_error", "friedman_mse"],
}

modele_base = RandomForestRegressor(random_state=42)


# Recherche des meilleurs paramètres
recherche = RandomizedSearchCV(
    estimator=modele_base,
    param_distributions=grille_parametres,
    n_iter=50,  # Test 50 combinaisons différentes
    cv=3,  # Validation croisée
    verbose=2,  # Pour print dans console
    random_state=42,
    n_jobs=-1,  # pour accélérer processus
)

# On lance la recherche des meilleurs paramètres
recherche.fit(X_train, y_train)

# On récupère celui avec meilleure performance
model = recherche.best_estimator_
model.fit(X_train, y_train)

# Prédiction sur données de test
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE : ", mae)
print("RMSE : ", rmse)
print("R2 : ", r2)


# VARIABLES LES PLUS IMPORTANTES DANS LE MODÈLE
importances = model.feature_importances_
colonnes = X.columns
df_importance = pd.DataFrame({"Variable": colonnes, "Importance": importances})

# On arrange de la plus importante à la moins importante et on garde juste les 15 premières
df_importance = df_importance.sort_values(by="Importance", ascending=False).head(15)

# On crée graphique
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Variable", data=df_importance)

plt.title("15 variables les plus importantes", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Variables", fontsize=12)
plt.show()


# ============================================================
# ANALYSE DE LA PRÉCISION PAR INTERVALLES
# ============================================================
erreurs_absolues = np.abs(y_test - y_pred)

pct_plus_ou_moins_3 = (erreurs_absolues <= 3).mean() * 100
pct_plus_ou_moins_5 = (erreurs_absolues <= 5).mean() * 100
pct_plus_ou_moins_10 = (erreurs_absolues <= 10).mean() * 100
pct_plus_ou_moins_16 = (erreurs_absolues <= 16).mean() * 100

print(
    f"\nProportion de prédictions exactes à +/- 3 rangs : {pct_plus_ou_moins_3:.2f} %"
)
print(f"Proportion de prédictions exactes à +/- 5 rangs : {pct_plus_ou_moins_5:.2f} %")
print(
    f"Proportion de prédictions exactes à +/- 10 rangs : {pct_plus_ou_moins_10:.2f} %"
)
print(
    f"Proportion de prédictions exactes à +/- 16 rangs : {pct_plus_ou_moins_16:.2f} %"
)
print("============================================================\n")


# ============================================================
# GRAPHIQUE : VRAIS RANGS VS PRÉDICTIONS
# ============================================================

plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="dodgerblue", edgecolor="black")

# Ajout de la droite y=x (le modèle parfait rang prédit = vrai rang)
plt.plot(
    [1, 64],
    [1, 64],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Prédiction parfaite (Erreur = 0)",
)

plt.xlim(0, 65)
plt.ylim(0, 65)
plt.title("Vrais rangs vs rangs prédits", fontsize=14, fontweight="bold")
plt.xlabel("Vrai rang au repêchage", fontsize=12)
plt.ylabel("Rang prédit", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
