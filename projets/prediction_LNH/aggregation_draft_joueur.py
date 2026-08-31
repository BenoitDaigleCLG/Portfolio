import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = "../Data/statistiques.xlsx"
OUTPUT_PATH = "../Data/dataset_joueur_draft.xlsx"


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

print("Nb joueurs ambigus retirés :", len(ambiguous_ids))
print("Shape après nettoyage :", df_clean.shape)


# ============================================================
# 3. FEATURES PAR SAISON
# ============================================================
df_clean["DATE_OF_BIRTH"] = pd.to_datetime(df_clean["DATE_OF_BIRTH"], errors="coerce")
df_clean["AGE_AT_SEASON"] = (
    df_clean["LEAGUE_YEAR_START"] - df_clean["DATE_OF_BIRTH"].dt.year
)

# évite division par zéro
height_m = df_clean["HEIGHT_CM"] / 100
df_clean["BMI_PROXY"] = np.where(
    height_m > 0, df_clean["WEIGHT_KG"] / (height_m**2), np.nan
)

gp_safe = df_clean["GP"].replace(0, np.nan)
df_clean["P_PER_GP"] = df_clean["P"] / gp_safe
df_clean["G_PER_GP"] = df_clean["G"] / gp_safe
df_clean["A_PER_GP"] = df_clean["A"] / gp_safe
df_clean["PIM_PER_GP"] = df_clean["PIM"] / gp_safe

df_clean = df_clean.sort_values(["PLAYER_ID", "LEAGUE_YEAR_START"])


# ============================================================
# 4. AGREGATION AU NIVEAU JOUEUR
#    1 ligne = 1 joueur
# ============================================================
df_player = (
    df_clean.groupby("PLAYER_ID")
    .agg(
        DRAFT_OVERALL=("DRAFT_OVERALL", "first"),
        DRAFT_YEAR=("DRAFT_YEAR", "first"),
        PRIMARY_POS=("PRIMARY_POS", first_mode),
        SECONDARY_POS=("SECONDARY_POS", first_mode),
        NATIONALITY=("NATIONALITY", first_mode),
        SHOOTS=("SHOOTS", first_mode),
        HEIGHT_CM=("HEIGHT_CM", "first"),
        WEIGHT_KG=("WEIGHT_KG", "first"),
        BMI_PROXY=("BMI_PROXY", "first"),
        AGE_LAST_SEASON=("AGE_AT_SEASON", "last"),
        N_PRE_DRAFT_SEASONS=("LEAGUE_YEAR_START", "nunique"),
        N_LEAGUES=("LEAGUE", "nunique"),
        TOTAL_GP=("GP", "sum"),
        TOTAL_G=("G", "sum"),
        TOTAL_A=("A", "sum"),
        TOTAL_P=("P", "sum"),
        TOTAL_PIM=("PIM", "sum"),
        MEAN_PPG=("PPG", "mean"),
        MAX_PPG=("PPG", "max"),
        LAST_GP=("GP", "last"),
        LAST_G=("G", "last"),
        LAST_A=("A", "last"),
        LAST_P=("P", "last"),
        LAST_PPG=("PPG", "last"),
        LAST_PIM=("PIM", "last"),
        LAST_PLUS_MINUS=(
            ("+/-", "last") if "+/-" in df_clean.columns else ("PIM", "last")
        ),
        LAST_LEAGUE=("LEAGUE", "last"),
        LAST_SEASON=("LEAGUE_YEAR_START", "last"),
        LAST_P_PER_GP=("P_PER_GP", "last"),
        LAST_G_PER_GP=("G_PER_GP", "last"),
        LAST_A_PER_GP=("A_PER_GP", "last"),
        LAST_PIM_PER_GP=("PIM_PER_GP", "last"),
    )
    .reset_index()
)

# progression entre dernière et avant-dernière saison
delta_ppg = (
    df_clean.groupby("PLAYER_ID")["PPG"]
    .apply(lambda s: s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan)
    .rename("DELTA_PPG_LAST_PREV")
    .reset_index()
)

df_player = df_player.merge(delta_ppg, on="PLAYER_ID", how="left")

print("Shape dataset joueur :", df_player.shape)
print(df_player.head())


# ============================================================
# 5. EXPORT
# ============================================================
df_player.to_excel(OUTPUT_PATH, index=False)
print(f"\nDataset agrégé exporté vers : {OUTPUT_PATH}")
