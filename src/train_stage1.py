import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from config import FEATURES, DATASET_PATH, MODELS_DIR
from preprocessing import clean_features
from evaluate import evaluate_model

df = pd.read_csv(DATASET_PATH)

X = clean_features(df, FEATURES)
y = df["Label"].astype("int8")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========================
# IMPROVED MODELS FOR BETTER CONFIDENCE
# ========================

# Enhanced Random Forest with higher probability calibration
rf = RandomForestClassifier(
    n_estimators=200,  # Increased from 50
    max_depth=15,  # Added depth control
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    oob_score=True  # For better probability estimates
)

# Enhanced XGBoost with calibration for sharp predictions
xgb = XGBClassifier(
    n_estimators=150,  # Increased from 100
    max_depth=6,
    learning_rate=0.1,  # Slightly increased from 0.05
    tree_method="hist",
    eval_metric="logloss",
    scale_pos_weight=1,  # Adjusted for class balance
    gamma=1,  # Increased to prevent overfitting
    random_state=42
)

# Gradient Boosting for additional ensemble strength
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    verbose=0
)

print("Training Stage 1 models...")
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
gb.fit(X_train, y_train)

joblib.dump(rf, MODELS_DIR / "stage1_rf.pkl")
joblib.dump(xgb, MODELS_DIR / "stage1_xgb.pkl")
joblib.dump(gb, MODELS_DIR / "stage1_gb.pkl")

print("Stage 1 models trained and saved")

evaluate_model(rf, X_test, y_test, model_name="Random Forest — Stage 1", binary=True)
evaluate_model(xgb, X_test, y_test, model_name="XGBoost — Stage 1", binary=True)
evaluate_model(gb, X_test, y_test, model_name="Gradient Boosting — Stage 1", binary=True)

# ========================
# ENSEMBLE VOTING FOR BETTER PREDICTIONS
# ========================
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
    voting='soft',  # Use probability averaging
    n_jobs=-1
)

ensemble.fit(X_train, y_train)
joblib.dump(ensemble, MODELS_DIR / "stage1_ensemble.pkl")

print("\nEnsemble model trained and saved")
evaluate_model(ensemble, X_test, y_test, model_name="Voting Ensemble — Stage 1", binary=True)