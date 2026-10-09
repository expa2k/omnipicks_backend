"""
OmniPicks — ML: Train All Models
Trains XGBoost models using the historical datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss
import joblib
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

def load_and_clean_data(sport: str):
    """Carga el CSV del deporte y prepara las variables X e y."""
    file_path = BASE_DIR / "datasets" / f"{sport}_history.csv"
    if not file_path.exists():
        logger.warning(f"Archivo no encontrado: {file_path}. Ejecuta download_datasets.py primero.")
        return None, None
        
    df = pd.read_csv(file_path)
    
    # 1. Filtramos datos posteriores a 2010 (últimos ~15 años)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year >= 2010]
        # Ordenamos cronológicamente para evitar fugas de información del futuro
        df = df.sort_values('date')
        
    # Limpiamos NaNs críticos
    df = df.dropna(subset=['score1', 'score2'])
    
    # 2. Definimos nuestras 'y' (Targets)
    # y_ml: 1 si gana Home (equipo 1), 0 si gana Away o Empatan
    df['home_win'] = (df['score1'] > df['score2']).astype(int)
    # y_total: Total de puntos anotados (para Over/Under)
    df['total_points'] = df['score1'] + df['score2']
    
    # 3. Extraemos las 'X' (Features - usamos ELO histórico)
    # FiveThirtyEight nos da los ELOs previos al partido
    features = ['elo1_pre', 'elo2_pre']
    if 'elo_prob1' in df.columns:
        features.extend(['elo_prob1', 'elo_prob2'])
        
    # Rellenamos NaNs en features numéricos
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(df[f].mean())
            
    X = df[features]
    
    return X, df

def train_models_for_sport(sport: str):
    """Entrena un modelo para predecir Ganador (Moneyline) y Totales (Over/Under)."""
    logger.info(f"--- Entrenando modelos para {sport.upper()} ---")
    
    X, df = load_and_clean_data(sport)
    if X is None:
        return
        
    # Variables objetivo
    y_win = df['home_win']
    y_total = df['total_points']
    
    # Split manual 80/20 (usamos datos viejos para entrenar, nuevos para testear)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_win_train, y_win_test = y_win.iloc[:split_idx], y_win.iloc[split_idx:]
    y_tot_train, y_tot_test = y_total.iloc[:split_idx], y_total.iloc[split_idx:]
    
    # 1. Entrenar Clasificador (¿Quién gana?)
    clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    logger.info("Entrenando clasificador Moneyline (XGBoost)...")
    clf.fit(X_train, y_win_train)
    
    # Evaluar clasificador
    win_preds = clf.predict(X_test)
    win_probs = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_win_test, win_preds)
    brier = brier_score_loss(y_win_test, win_probs)
    logger.info(f"✅ Exactitud (Accuracy): {acc:.2f} | Brier Score: {brier:.4f}")
    
    # 2. Entrenar Regresor (Total de Puntos para O/U)
    reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
    logger.info("Entrenando regresor Totales (XGBoost)...")
    reg.fit(X_train, y_tot_train)
    
    # Evaluar regresor
    tot_preds = reg.predict(X_test)
    mae = np.mean(np.abs(y_tot_test - tot_preds))
    logger.info(f"✅ Error Absoluto Medio (MAE) de puntos: {mae:.2f}")
    
    # 3. Guardar modelos
    save_dir = BASE_DIR / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    clf_path = save_dir / f"{sport}_moneyline_xgb.joblib"
    reg_path = save_dir / f"{sport}_totals_xgb.joblib"
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    logger.info(f"💾 Modelos guardados en {save_dir}/\n")

def main():
    sports = ["nba", "nfl", "mlb", "soccer"]
    for s in sports:
        train_models_for_sport(s)
    logger.info("🎉 Entrenamiento de todos los modelos finalizado exitosamente.")

if __name__ == "__main__":
    main()
