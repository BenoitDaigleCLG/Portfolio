import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ouverture du dataset et séparation en groupes de draft
INPUT_PATH = "../Data/dataset_joueur_draft.xlsx"
df = pd.read_excel(INPUT_PATH)
bins = [
    0,
    32,
    100,
    df["DRAFT_OVERALL"].max(),
]  # Découpage en 3 groupes selon la logique du draft NHL :
# Tour 1 (Top 32), Tours 2-3 (33-100), Tours 4+ (> 100)
labels = ["Top 32", "33–100", "> 100"]

df["DRAFT_GROUP"] = pd.cut(
    df["DRAFT_OVERALL"], bins=bins, labels=labels, include_lowest=True
)


# Sélection des variables quantitatives pour l'ACP
quant_vars_selected = [
    # Production offensive — non redondant
    "LAST_P_PER_GP",  # F=94  (on a retiré LAST_PPG, identique)
    "LAST_G_PER_GP",  # F=59
    "LAST_A_PER_GP",  # F=99  (on a retiré LAST_A, LAST_P)
    "LAST_PLUS_MINUS",  # F=42
    "DELTA_PPG_LAST_PREV",  # F=23
    # Carrière pré-draft
    "MEAN_PPG",  # F=67  (on a retiré MAX_PPG, corrélé)
    "TOTAL_P",  # F=62  (on a retiré TOTAL_G, TOTAL_A)
    # Contexte
    "AGE_LAST_SEASON",  # F=86
    "LAST_GP",  # F=21
    # Physique
    "HEIGHT_CM",  # F=28
    "WEIGHT_KG",  # F=45  (on a retiré BMI_PROXY, dérivé)
]


# Imputation des valeurs manquantes par la médiane (plus robuste que la moyenne)
# puis standardisation : moyenne 0, écart-type 1, requis pour l'ACP
X = df[quant_vars_selected].fillna(df[quant_vars_selected].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(
    n_components=len(quant_vars_selected)
)  # ACP sur toutes les composantes pour choisir le nombre optimal via le scree plot
X_pca = pca.fit_transform(X_scaled)


explained_variance = np.cumsum(pca.explained_variance_ratio_)


scores = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
scores["DRAFT_GROUP"] = df["DRAFT_GROUP"].values
loadings = pd.DataFrame(
    pca.components_.T,
    index=quant_vars_selected,
    columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
)


# ============================================================
# FIGURE ACP – Scree plot, projection, cercle de corrélation
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 5))

# ------------------------------------------------------------
# (1) Scree plot – variance expliquée cumulée
# ------------------------------------------------------------
axes[0, 0].plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
axes[0, 0].set_xlabel("Nombre de composantes")
axes[0, 0].set_ylabel("Variance expliquée cumulée")
axes[0, 0].set_title("Scree plot – variance expliquée cumulée")

# ------------------------------------------------------------
# (2) Projection ACP – PC1 vs PC2
# ------------------------------------------------------------
for group in scores["DRAFT_GROUP"].unique():
    subset = scores[scores["DRAFT_GROUP"] == group]
    axes[0, 1].scatter(subset["PC1"], subset["PC2"], alpha=0.6, label=group)

axes[0, 1].set_xlabel("PC1")
axes[0, 1].set_ylabel("PC2")
axes[0, 1].set_title("Projection ACP (PC1 vs PC2)")
axes[0, 1].legend(title="Rang de draft")

# ------------------------------------------------------------
# (3) Cercle de corrélation – PC1 vs PC2
# ------------------------------------------------------------
for var in loadings.index:
    axes[1, 0].arrow(
        0,
        0,
        loadings.loc[var, "PC1"],
        loadings.loc[var, "PC2"],
        head_width=0.02,
        alpha=0.6,
    )
    axes[1, 0].text(
        loadings.loc[var, "PC1"] * 1.1, loadings.loc[var, "PC2"] * 1.1, var, fontsize=8
    )

axes[1, 0].axhline(0, color="black", linewidth=0.5)
axes[1, 0].axvline(0, color="black", linewidth=0.5)
axes[1, 0].set_xlabel("PC1")
axes[1, 0].set_ylabel("PC2")
axes[1, 0].set_title("Cercle de corrélation (PC1 vs PC2)")


# -------------------------------------------------------------
# (4) Cercle de corrélation – PC1 vs PC3
# -------------------------------------------------------------
for var in loadings.index:
    axes[1, 1].arrow(
        0,
        0,
        loadings.loc[var, "PC1"],
        loadings.loc[var, "PC3"],
        head_width=0.02,
        alpha=0.6,
    )
    axes[1, 1].text(
        loadings.loc[var, "PC1"] * 1.1, loadings.loc[var, "PC3"] * 1.1, var, fontsize=8
    )

axes[1, 1].axhline(0, color="black", linewidth=0.5)
axes[1, 1].axvline(0, color="black", linewidth=0.5)
axes[1, 1].set_xlabel("PC1")
axes[1, 1].set_ylabel("PC3")
axes[1, 1].set_title("Cercle de corrélation (PC1 vs PC3)")

plt.tight_layout()
plt.show()


# -------------------------------------------------------------
# Figure distribution des PC par groupe de draft
# -------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, pc in zip(axes, ["PC1", "PC2", "PC3"]):
    for group in ["Top 32", "33–100", "> 100"]:
        subset = scores[scores["DRAFT_GROUP"] == group]
        sns.kdeplot(subset[pc], ax=ax, label=group, fill=True, alpha=0.3)
    ax.set_title(f"Distribution {pc} par groupe")
    ax.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# PC1 vs PC3 —
# -------------------------------------------------------------
plt.figure()
for group in scores["DRAFT_GROUP"].unique():
    subset = scores[scores["DRAFT_GROUP"] == group]
    plt.scatter(subset["PC1"], subset["PC3"], alpha=0.4, label=group)
plt.xlabel("PC1")
plt.ylabel("PC3")
plt.title("ACP – PC1 vs PC3")
plt.legend(title="Rang de draft")
plt.show()


# --------------------------------------------------------------
# Test pour savoir si les moyennes PC1 diffèrent significativement
# entre les 3 groupes de draft (H0 : moyennes égales)
# --------------------------------------------------------------
top32_pc1 = scores[scores["DRAFT_GROUP"] == "Top 32"]["PC1"]
mid_pc1 = scores[scores["DRAFT_GROUP"] == "33–100"]["PC1"]
low_pc1 = scores[scores["DRAFT_GROUP"] == "> 100"]["PC1"]

f_stat, p_val = stats.f_oneway(top32_pc1, mid_pc1, low_pc1)
print(f"ANOVA sur PC1 : F={f_stat:.2f}, p={p_val:.2e}")


# Comparer les moyennes PC1 par groupe
print(scores.groupby("DRAFT_GROUP")["PC1"].agg(["mean", "std", "count"]))
