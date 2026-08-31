## ============================================================
## 0. Chargement des librairies
## ============================================================
# À exécuter une seule fois au besoin :
# install.packages(c("BatchGetSymbols","data.table","TTR","quantmod","zoo","FactoMineR","factoextra"))

library(quantmod)      # Pour getSymbols, objets xts, etc.
library(zoo)           # Pour la gestion de séries temporelles et na.locf
library(BatchGetSymbols) # Pour télécharger facilement des données boursières
library(data.table)    # Pour manipuler les données de façon efficace
library(TTR)           # Pour les indicateurs techniques (RSI, ATR, SMA, etc.)
library(FactoMineR)    # Pour la PCA (ACP)
library(factoextra)    # Pour la visualisation des résultats de PCA

## ============================================================
## 1. Données S&P 500 (Yahoo + BatchGetSymbols) + indicateurs
## ============================================================

ticker     <- "^GSPC"  # Symbole de l'indice S&P500 sur Yahoo Finance
first_date <- Sys.Date() - round(365.25 * 20)  # Environ 20 ans d’historique
last_date  <- Sys.Date()
cache_dir  <- "BGS_cache"                       # Dossier de cache pour BatchGetSymbols

# Téléchargement des données via BatchGetSymbols
res <- BatchGetSymbols(
  tickers      = ticker,
  first.date   = first_date,
  last.date    = last_date,
  freq.data    = "daily",      # Données quotidiennes
  cache.folder = cache_dir     # Utilise un cache local pour éviter de retélécharger
)

# Construction du tableau principal des prix (sous forme de data.table)
dt <- as.data.table(res$df.tickers)[
  , .(
    Date   = ref.date,        # Date de l'observation
    Open   = price.open,      # Prix d'ouverture
    High   = price.high,      # Plus haut de la journée
    Low    = price.low,       # Plus bas de la journée
    Close  = price.close,     # Prix de clôture
    Volume = volume,          # Volume échangé
    Price  = price.adjusted   # Prix ajusté (dividendes, splits, etc.)
  )
][order(Date)]                # On s’assure que les données sont triées par date

# Conversion des colonnes utiles en objets xts pour les indicateurs techniques
HLC_xts    <- xts(cbind(High = dt$High, Low = dt$Low, Close = dt$Close),
                  order.by = dt$Date)
Price_xts  <- xts(dt$Price,  order.by = dt$Date)
Volume_xts <- xts(dt$Volume, order.by = dt$Date)

## ------------------------------------------------------------
## 1.1 Indicateurs techniques (TTR)
## ------------------------------------------------------------

# Moyenne mobile à 200 jours
#dt[, SMA_200       := as.numeric(SMA(Price_xts, n = 200))]

# RSI sur 14 jours
dt[, RSI_14        := as.numeric(RSI(Price_xts, n = 14))]

# Momentum sur 10 jours
dt[, Momentum_10   := as.numeric(momentum(Price_xts, n = 10))]

# Average True Range sur 14 jours (mesure de volatilité)
dt[, ATR_14        := as.numeric(ATR(HLC_xts, n = 14)[, "atr"])]

# Ecart-type roulant sur 20 jours (volatilité historique)
dt[, RollingStd_20 := as.numeric(runSD(Price_xts, n = 20))]

# On-Balance Volume (OBV) : volume cumulé pondéré par le signe du rendement
dt[, OBV           := as.numeric(OBV(Price_xts, Volume_xts))]

# Chaikin Money Flow sur 20 jours (liquidité / pression à l'achat/vente)
dt[, CMF_20        := as.numeric(CMF(HLC_xts, Volume_xts, n = 20))]

# Money Flow Index sur 14 jours (RSI version volume / flux monétaire)
dt[, MFI_14        := as.numeric(MFI(HLC_xts, Volume_xts, n = 14))]

# Commodity Channel Index sur 20 jours (déviation par rapport à la moyenne)
dt[, CCI_20        := as.numeric(CCI(HLC_xts, n = 20))]

## ------------------------------------------------------------
## 1.2 Rendements multi-horizons (stationnaires)
## ------------------------------------------------------------

# Approximation en jours de bourse :
#   5 jours  ≈ 1 semaine
#   21 jours ≈ 1 mois
#   252 jours ≈ 1 année
dt[, week_return  := Price / shift(Price,  5)  - 1]  # Rendement ≈ 1 semaine
dt[, month_return := Price / shift(Price, 21) - 1]  # Rendement ≈ 1 mois
dt[, year_return  := Price / shift(Price, 252) - 1] # Rendement ≈ 1 an

## ------------------------------------------------------------
## 1.3 Ratios prix / moyennes mobiles (stationnaires)
## ------------------------------------------------------------

# Ratio entre le prix et la SMA 20 jours
dt[, ratio_price_sma20  := Price / SMA(Price,  20)]

# Ratio entre le prix et la SMA 200 jours
dt[, ratio_price_sma200 := Price / SMA(Price, 200)]

## ============================================================
## 2. Macro : spreads, VIX, or, pétrole
## ============================================================

## ------------------------------------------------------------
## 2.1 Spread de taux : 10 ans – 2 ans (FRED)
## ------------------------------------------------------------

getSymbols(c("DGS10", "DGS2"),
           src  = "FRED",
           from = first_date,
           to   = last_date)

# Spread de taux 10 ans – 2 ans (en points de pourcentage)
TS_10y_2y <- DGS10 - DGS2
colnames(TS_10y_2y) <- "TermSpread_10y_2y"

## ------------------------------------------------------------
## 2.2 Spread de crédit : IG – GOV (en points de base)
##     IG = OAS corporatif BBB (bps)
##     GOV = taux 10 ans gouvernemental, converti en bps
## ------------------------------------------------------------

# IG OAS (BBB) en bps (spread de crédit investment grade)
getSymbols("BAMLC0A4CBBB",
           src  = "FRED",
           from = first_date,
           to   = last_date)

IG_OAS <- BAMLC0A4CBBB
colnames(IG_OAS) <- "IG_OAS_bps"

# Taux 10 ans (DGS10) est déjà téléchargé plus haut
# On le convertit en points de base pour avoir la même unité que IG_OAS
DGS10_bps <- DGS10 * 100
colnames(DGS10_bps) <- "GOV_10y_bps"

# Spread IG – GOV (en bps) = spread de crédit net du taux sans risque
CreditSpread_IG_GOV <- IG_OAS - DGS10_bps
colnames(CreditSpread_IG_GOV) <- "CreditSpread_IG_GOV_bps"

## ------------------------------------------------------------
## 2.3 VIX, or, pétrole (Yahoo Finance)
## ------------------------------------------------------------

getSymbols(c("^VIX", "GC=F", "CL=F"),
           src  = "yahoo",
           from = first_date,
           to   = last_date)

# VIX en clôture
VIX_close  <- Cl(VIX)
colnames(VIX_close)  <- "VIX_close"

# Prix de l'or (futures GC=F) en USD
Gold_price <- Ad(`GC=F`)
colnames(Gold_price) <- "Gold_price_usd"

# Prix du pétrole WTI (futures CL=F) en USD
Oil_price  <- Ad(`CL=F`)
colnames(Oil_price)  <- "Oil_price_usd"

## ============================================================
## 3. Calcul des variations de spreads (1 jour, 1 mois)
## ============================================================

# 1 jour = 1; 1 mois ≈ 21 jours de bourse
horizons <- list(
  "1d" = 1,
  "1m" = 21
)

# Fonction générique pour créer des colonnes de variation multi-horizon
make_horizon_changes <- function(x, base_name) {
  out_list <- vector("list", length(horizons))
  names(out_list) <- names(horizons)
  
  for (h_name in names(horizons)) {
    k   <- horizons[[h_name]]
    tmp <- x - lag(x, k)                           # Δ spread = spread_t - spread_{t-k}
    colnames(tmp) <- paste0(base_name, "_chg_", h_name)  # Nom de colonne : base_chg_1d / base_chg_1m
    out_list[[h_name]] <- tmp
  }
  
  do.call(merge, out_list)   # On fusionne toutes les colonnes en un seul objet xts
}

# Variations du spread de taux 10y–2y
TS_changes <- make_horizon_changes(
  TS_10y_2y,
  "TermSpread_10y_2y"
)

# Variations du spread de crédit IG–GOV
CS_changes <- make_horizon_changes(
  CreditSpread_IG_GOV,
  "CreditSpread_IG_GOV_bps"
)

## ============================================================
## 4. Dataset macro final : VIX, or, pétrole + variations de spreads
## ============================================================

# On fusionne seulement :
# - VIX, Gold, Oil (en niveau)
# - Variations de TermSpread_10y_2y (1j, 1m)
# - Variations de CreditSpread_IG_GOV_bps (1j, 1m)
macro_all <- merge(
  VIX_close,
  Gold_price,
  Oil_price,
  TS_changes,
  CS_changes
)

# On supprime les premières lignes avec NA dues aux lags (21 jours)
macro_all <- macro_all[complete.cases(macro_all), ]

## ============================================================
## 5. Fusion finale avec le dataset S&P 500
## ============================================================

# Conversion xts -> data.table, avec une colonne Date explcite
macro_dt <- as.data.table(macro_all, keep.rownames = "Date")
macro_dt[, Date := as.Date(Date)]

# Fusion par la date (left join : on garde toutes les dates du S&P500)
setkey(macro_dt, Date)
setkey(dt,      Date)

dt <- merge(dt, macro_dt, all.x = TRUE)

## ------------------------------------------------------------
## 5.1 Sauvegarde brute avec indicateurs (avant nettoyage)
## ------------------------------------------------------------

fwrite(dt, "sp500_20y_daily_with_indicators.csv")

## ============================================================
## 6. Nettoyage des valeurs extrêmes et des NA
## ============================================================

# Colonnes à exclure des transformations (id / prix bruts)
drop_cols <- c("Date","Open","High","Low","Close","Price")

# Ensemble des colonnes à traiter comme indicateurs/features
indicators <- setdiff(names(dt), drop_cols)

# Fonction pour tronquer les valeurs extrêmes aux 1er et 99e percentiles
clean_extreme_values <- function(values, lower=0.01, upper=0.99) {
  qs <- quantile(values, probs = c(lower, upper), na.rm = TRUE)
  # On écrase les valeurs en-dessous du 1er percentile
  values[values < qs[1]] <- qs[1]
  # On écrase les valeurs au-dessus du 99e percentile
  values[values > qs[2]] <- qs[2]
  return(values)
}

# Application de la fonction à toutes les colonnes indicateurs
dt[, (indicators) := lapply(.SD, clean_extreme_values), .SDcols = indicators]

# On enlève toutes les lignes où il manque au moins une valeur parmi les indicateurs
dt <- dt[complete.cases(dt[, indicators, with = FALSE]), ]

## ------------------------------------------------------------
## 6.1 Sauvegarde du dataset nettoyé
## ------------------------------------------------------------

fwrite(dt, "sp500_20y_daily_clean.csv")

## ============================================================
## 7. Analyse en composantes principales (PCA/ACP)
## ============================================================

# On extrait uniquement les colonnes indicateurs dans un data.frame (FactoMineR attend un data.frame)
df <- as.data.frame(dt[, ..indicators])

# PCA avec mise à l'échelle (scale.unit=TRUE), on demande 20 composantes
pca_scaled <- PCA(df, scale.unit = TRUE, ncp = 20, graph = FALSE)

# Visualisation du scree plot (variance expliquée par composante)
fviz_eig(pca_scaled)

# Récupération des valeurs propres (variance de chaque axe)
get_eigenvalue(pca_scaled)

# Contribution des variables à chaque axe (arrondie à 2 décimales)
contrib_rounded <- as.data.frame(round(pca_scaled$var$contrib, 2))
contrib_rounded

# Visualisation des contributions des variables aux 5 premiers axes principaux
fviz_contrib(pca_scaled, choice = "var", axes = 1)
fviz_contrib(pca_scaled, choice = "var", axes = 2)
fviz_contrib(pca_scaled, choice = "var", axes = 3)
fviz_contrib(pca_scaled, choice = "var", axes = 4)
fviz_contrib(pca_scaled, choice = "var", axes = 5)

# Extract cos² values for variables on axes 1 and 2
cos2_mat <- pca_scaled$var$cos2

# Compute combined cos² on axes 3 and 4
cos2_PC12 <- cos2_mat[,1] + cos2_mat[,2]

# Keep only variables with cos² > 0.25
vars_keep12 <- names(cos2_PC12[cos2_PC12 > 0.15])
vars_keep12


fviz_pca_var(
  pca_scaled,
  axes = c(1, 2),                       # Plot PC3 vs PC4
  col.var = "cos2",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE,
  select.var = list(name = vars_keep12)   # <-- keep only cos2 > 0.25
)


# Compute combined cos² on axes 3 and 4
cos2_PC34 <- cos2_mat[,3] + cos2_mat[,4]

# Keep only variables with cos² > 0.25
vars_keep34 <- names(cos2_PC34[cos2_PC34 > 0.15])
vars_keep34

fviz_pca_var(
  pca_scaled,
  axes = c(3, 4),                       # Plot PC3 vs PC4
  col.var = "cos2",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE,
  select.var = list(name = vars_keep34)   # <-- keep only cos2 > 0.25
)


## ============================================================
## 8. Calcule des corrélations entre les indicateurs
## ============================================================

correlation_matrix <- cor(df)

## ============================================================
## 9. Sélection d’un sous-ensemble d’indicateurs après ACP
## ============================================================

# Liste d'indicateurs retenus (par exemple, ceux qui semblent les plus informatifs)
# NB : j'ai corrigé ici "Momentum" -> "Momentum_10" pour correspondre au nom de la colonne
indicateurs_retenus <- c(
  "Volume",           
  "RSI_14",    
  "ATR_14",            
  "OBV",            
  "ratio_price_sma200",
  "VIX_close",
  "TermSpread_10y_2y_chg_1d",
  "CreditSpread_IG_GOV_bps_chg_1d",
  "TermSpread_10y_2y_chg_1m",
  "CreditSpread_IG_GOV_bps_chg_1m"
)

# Colonnes finales à garder (Date, Price + indicateurs sélectionnés)
colonnes_a_garder <- c("Date", "Price", indicateurs_retenus)

# Sous-échantillon final de dt
dt_temp <- dt[, ..colonnes_a_garder]

# Sauvegarde du dataset "réduit" après ACP
fwrite(dt_temp, "sp500_20y_daily_after_ACP.csv")
