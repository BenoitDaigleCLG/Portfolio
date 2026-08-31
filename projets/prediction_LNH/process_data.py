import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = "../Data/statistiques.xlsx"
OUTPUT_PATH = "../Data/randomforest.xlsx"


# ============================================================
# OUTILS
# ============================================================
def first_mode(series: pd.Series):
    """
    Retourne le premier mode d'une série.

    Retourne np.nan si aucun mode n'est trouvé.
    """
    mode_vals = series.mode(dropna=True)
    return mode_vals.iloc[0] if len(mode_vals) > 0 else np.nan


# ============================================================
# 1. CHARGEMENT
# ============================================================
df = pd.read_excel(INPUT_PATH)

print("Shape initiale :", df.shape)
print("Nb joueurs distincts :", df["PLAYER_ID"].nunique())


# ============================================================
# 2. NETTOYAGE DES JOUEURS AMBIGUS
#    On retire les joueurs qui ont plusieurs DRAFT_YEAR
#    ou plusieurs DRAFT_OVERALL
# ============================================================
target_check = df.groupby("PLAYER_ID")[["DRAFT_OVERALL", "DRAFT_YEAR"]].nunique()
ambiguous_ids = target_check[
    (target_check["DRAFT_OVERALL"] > 1) | (target_check["DRAFT_YEAR"] > 1)
].index

df_clean = df[~df["PLAYER_ID"].isin(ambiguous_ids)].copy()

print("Nombre de joueurs ambigus retirés :", len(ambiguous_ids))
print("Nombre lignes et colonnes après nettoyage :", df_clean.shape)


# ============================================================
# 3. FEATURES PAR SAISON
# ============================================================
# On met dans bon format la date pour calculs
df_clean["DATE_OF_BIRTH"] = pd.to_datetime(df_clean["DATE_OF_BIRTH"], errors="coerce")
# Age pour chaque saison
df_clean["AGE_AT_SEASON"] = (
    df_clean["LEAGUE_YEAR_START"] - df_clean["DATE_OF_BIRTH"].dt.year
)

# Taille en mètres
height_m = df_clean["HEIGHT_CM"] / 100
# Indice qui lie poids et taille
df_clean["BMI_PROXY"] = np.where(
    height_m > 0, df_clean["WEIGHT_KG"] / (height_m**2), np.nan
)

gp_safe_clean = df_clean["GP"].replace(0, np.nan)
# Buts, assists et minutes de pénalité par game (on oublie pas qu'ici c'est encore par saison)
df_clean["G_PER_GP"] = df_clean["G"] / gp_safe_clean
df_clean["A_PER_GP"] = df_clean["A"] / gp_safe_clean
df_clean["PIM_PER_GP"] = df_clean["PIM"] / gp_safe_clean
df_clean = df_clean.sort_values(["PLAYER_ID", "LEAGUE_YEAR_START", "GP"])


# ============================================================
# 4. AGREGATION AU NIVEAU JOUEUR
#    1 ligne = 1 joueur
# ============================================================

# On fusionne les statistiques des ligues jouées dans la même année
# Comme ça un joueur qui a fait plusieur ligues dans la même année va avoir ses stats ensemble
df_seasons = (
    df_clean.groupby(["PLAYER_ID", "LEAGUE_YEAR_START"])
    .agg(
        LAST_GP=("GP", "sum"),
        LAST_G=("G", "sum"),
        LAST_A=("A", "sum"),
        LAST_P=("P", "sum"),
        LAST_PIM=("PIM", "sum"),
        LAST_LEAGUE=("LEAGUE", "last"),
    )
    .reset_index()
)

# On recalcule les statistiques maintenant qu'on a bien les bonnes
# années pré-repêchage
gp_safe = df_seasons["LAST_GP"].replace(0, np.nan)
df_seasons["LAST_P_PER_GP"] = df_seasons["LAST_P"] / gp_safe
df_seasons["LAST_G_PER_GP"] = df_seasons["LAST_G"] / gp_safe
df_seasons["LAST_A_PER_GP"] = df_seasons["LAST_A"] / gp_safe
df_seasons["LAST_PIM_PER_GP"] = df_seasons["LAST_PIM"] / gp_safe
df_seasons["LAST_PPG"] = df_seasons["LAST_P_PER_GP"]

# On va tout simplement garder la dernière année de chaque joueur
# avant son repêchage
df_seasons = df_seasons.rename(columns={"LEAGUE_YEAR_START": "LAST_SEASON"})
df_last_season = df_seasons.sort_values(["PLAYER_ID", "LAST_SEASON"])

# Ce df contient bien les statistiques de la dernière saison de chaque joueur
df_last_season = df_last_season.groupby("PLAYER_ID").last().reset_index()

# On vient regrouper les statistiques pour avoir une ligne par joueur
df_player = (
    df_clean.groupby("PLAYER_ID")
    .agg(
        DRAFT_OVERALL=("DRAFT_OVERALL", "first"),
        DRAFT_YEAR=("DRAFT_YEAR", "first"),
        DATE_OF_BIRTH=("DATE_OF_BIRTH", "first"),
        PRIMARY_POS=("PRIMARY_POS", first_mode),
        NATIONALITY=("NATIONALITY", first_mode),
        SHOOTS=("SHOOTS", first_mode),
        HEIGHT_CM=("HEIGHT_CM", "last"),
        WEIGHT_KG=("WEIGHT_KG", "last"),
        BMI_PROXY=("BMI_PROXY", "last"),
        AGE_LAST_SEASON=("AGE_AT_SEASON", "last"),
        TOTAL_GP=("GP", "sum"),
        TOTAL_G=("G", "sum"),
        TOTAL_A=("A", "sum"),
        TOTAL_P=("P", "sum"),
        TOTAL_PIM=("PIM", "sum"),
        MAX_PPG=("PPG", "max"),
    )
    .reset_index()
)

# On merge la table initiale df_player avec les statistiques calculées de sa dernière saison
df_player = df_player.merge(df_last_season, on="PLAYER_ID", how="left")
print(df_player.columns)

# On calcule les nouvelles statistiques en carrière pour ajouter à df_player
gp_safe_career = df_player["TOTAL_GP"].replace(0, np.nan)
df_player["GPG"] = df_player["TOTAL_G"] / gp_safe_career
df_player["APG"] = df_player["TOTAL_A"] / gp_safe_career
df_player["PPG"] = df_player["TOTAL_P"] / gp_safe_career
df_player["PIM_PGP"] = df_player["TOTAL_PIM"] / gp_safe_career

# Comment le joueur dans sa carrière a fait de buts par rapport à son nombre de points
df_player["SNIPER_RATIO"] = df_player["GPG"] / df_player["PPG"].replace(0, np.nan)
# COmment le joueur s'est amélioré dans sa dernière saison
df_player["MOMENTUM_PPG"] = df_player["LAST_PPG"] / df_player["PPG"].replace(0, np.nan)
# Comment le joueur dans sa carrière a eu d'assists par rapport à son nombre de points
df_player["PLAYMAKER_RATIO"] = df_player["APG"] / df_player["PPG"].replace(0, np.nan)

# Poids extraits de l'article de Patrick Bacon
# Ce n'est pas vrai que 60 points faits dans la KHL vaut 60 points fait dans la NCAA
# Niveau de difficulté pas le même donc on peut venir ajuster ça
nhle_weights = {
    "KHL": 0.80,  # Ligue russe
    "SHL": 0.58,  # Ligue suédoise
    "CZECH": 0.45,  # Extraliga tchèque
    "NLA": 0.43,  # Ligue suisse
    "LIIGA": 0.43,  # Ligue finlandaise
    "DEL": 0.38,  # Ligue allemande
    "NCAA": 0.33,  # Universitaire américain
    "OHL": 0.32,  # Junior Ontario
    "WHL": 0.29,  # Junior Ouest
    "USHL": 0.27,  # Junior Américain
    "QMJHL": 0.26,  # Junior Québec
    "BCHL": 0.17,  # Junior A (Colombie-Britannique)
    "AJHL": 0.15,  # Junior A (Alberta)
    "OJHL": 0.11,  # Junior A (Ontario)
    "NOJHL": 0.10,  # Junior A (Nord de l'Ontario)
    "USHS-MN": 0.09,  # High School (Minnesota)
    "USHS-PREP": 0.08,  # Prep School (USA)
}

# On mutliplie les points faits dans la dernière saison par leur poids respectifs
multiplicateur = df_player["LAST_LEAGUE"].map(nhle_weights).fillna(0.15)
df_player["NHLe_PPG"] = df_player["LAST_P_PER_GP"] * multiplicateur

# Pas mal toujours autour du 25 juin
draft_date = pd.to_datetime(
    df_player["DRAFT_YEAR"].fillna(2000).astype(int).astype(str) + "-06-25"
)
df_player["EXACT_AGE_AT_DRAFT"] = (
    draft_date - df_player["DATE_OF_BIRTH"]
).dt.days / 365.25

print("Shape dataset joueur :", df_player.shape)
df_propre = df_player.dropna()
print("Nouvelle shape dataset joueur :", df_player.shape)

# On va faire notre modèle jsute sur les deux premières rondes. Après ça devient trop difficile et plus aléatoire. Plus de bruit.
df_propre = df_propre[df_propre["DRAFT_OVERALL"] <= 64]
# On s'assure de garder des joueurs qui ont joué quelques matchs pour ne pas fausser le modèles avec des joueurs qui ont 0 matchs
df_propre = df_propre[df_propre["LAST_GP"] >= 5]

# On enlève les quelques cas mal annotées, car un Wing fait déjà partie de LW ou RW et Forward regroupe déjà tous les avants
# Donc maintenant on a bien Défenseur, centre, aile gauche et aile droite
df_propre = df_propre[~df_propre["PRIMARY_POS"].isin(["W", "F"])]

# ============================================================
# 5. EXPORT
# ============================================================
colonnes_finales = [
    "PLAYER_ID",
    "DRAFT_OVERALL",
    "DRAFT_YEAR",
    "PRIMARY_POS",
    "NATIONALITY",
    "SHOOTS",
    "WEIGHT_KG",
    "HEIGHT_CM",
    "BMI_PROXY",
    "TOTAL_GP",
    "GPG",
    "APG",
    "PPG",
    "PIM_PGP",
    "MAX_PPG",
    "LAST_GP",
    "LAST_LEAGUE",
    "SNIPER_RATIO",
    "LAST_G_PER_GP",
    "LAST_P_PER_GP",
    "LAST_A_PER_GP",
    "LAST_PIM_PER_GP",
    "MOMENTUM_PPG",
    "PLAYMAKER_RATIO",
    "EXACT_AGE_AT_DRAFT",
    "NHLe_PPG",
]

# On garde seulement les colonnes voulues
df_final = df_propre[colonnes_finales]

df_final.to_excel(OUTPUT_PATH, index=False)


# ============================================================
# 6. ANALYSE DES VARIABLES FINALES
# ============================================================
print("\n" + "=" * 50)
print("RAPPORT STATISTIQUE DU DATASET FINAL")
print("=" * 50)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Types des variables et valeurs manquantes
print("\n--- TYPES ET VALEURS MANQUANTES ---")
print(df_final.info())

# Nombre de classes de chaque variable
print("\n--- NOMBRE DE CLASSES ---")
print(df_final.nunique())

# Statistiques
print("\n--- STATISTIQUES NUMÉRIQUES ---")
print(df_final.describe().round(2))

# Statistiques pour variables non-numériques
print("\n--- STATISTIQUES CATÉGORIELLES ---")
colonnes_texte = df_final.select_dtypes(include=["object", "string"]).columns
if len(colonnes_texte) > 0:
    print(df_final[colonnes_texte].describe())
else:
    print("Aucune colonne de texte trouvée.")
