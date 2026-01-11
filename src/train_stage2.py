import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from config import FEATURES, DATASET_PATH, MODELS_DIR
from preprocessing import clean_features, encode_multiclass
from evaluate import evaluate_model

df = pd.read_csv(DATASET_PATH)
df = df[df["Label"] == 1]  # only attacks

X = clean_features(df, FEATURES)
y, encoder = encode_multiclass(df["Attack"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========================
# IMPROVED MODELS FOR BETTER CONFIDENCE
# ========================

# Enhanced Random Forest
rf = RandomForestClassifier(
    n_estimators=200,  # Increased from 50
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    oob_score=True
)

# Enhanced XGBoost with better parameters
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=7,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="mlogloss",
    gamma=1,
    scale_pos_weight=1,
    random_state=42
)

# Gradient Boosting for multi-class
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    verbose=0
)

print("Training Stage 2 models...")
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
gb.fit(X_train, y_train)

joblib.dump((rf, encoder), MODELS_DIR / "stage2_rf.pkl")
joblib.dump((xgb, encoder), MODELS_DIR / "stage2_xgb.pkl")
joblib.dump((gb, encoder), MODELS_DIR / "stage2_gb.pkl")

print("Stage 2 models trained and saved")

evaluate_model(rf, X_test, y_test, model_name="Random Forest — Stage 2", binary=False, encoder=encoder)
evaluate_model(xgb, X_test, y_test, model_name="XGBoost — Stage 2", binary=False, encoder=encoder)
evaluate_model(gb, X_test, y_test, model_name="Gradient Boosting — Stage 2", binary=False, encoder=encoder)

# ========================
# ENSEMBLE VOTING FOR ATTACK TYPE
# ========================
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
    voting='soft'
)

ensemble.fit(X_train, y_train)
joblib.dump((ensemble, encoder), MODELS_DIR / "stage2_ensemble.pkl")

print("\nEnsemble model trained and saved")
evaluate_model(ensemble, X_test, y_test, model_name="Voting Ensemble — Stage 2", binary=False, encoder=encoder)