import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, LocalOutlierFactor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report

from config import FEATURES, DATASET_PATH, MODELS_DIR
from preprocessing import clean_features


df = pd.read_csv(DATASET_PATH)

y = df["Label"].astype("int8")

X = clean_features(df, FEATURES)

X_benign = X[y == 0]

contamination_rate = y.mean()

# ========================
# IMPROVED ANOMALY DETECTION
# ========================

# Isolation Forest with better parameters for malicious detection
model = IsolationForest(
    n_estimators=250,  # Increased from 150
    contamination=contamination_rate,
    max_samples='auto',
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Training Isolation Forest anomaly detector...")
model.fit(X_benign)

joblib.dump(model, MODELS_DIR / "anomaly_iforest.pkl")

print("Isolation Forest trained and saved")

# ========================
# EVALUATION
# ========================

scores = -model.decision_function(X)

roc = roc_auc_score(y, scores)
precision, recall, thresholds = precision_recall_curve(y, scores)
pr_auc = auc(recall, precision)

print(f"\n✓ Anomaly ROC-AUC: {roc:.4f}")
print(f"✓ Anomaly PR-AUC : {pr_auc:.4f}")

threshold = np.percentile(scores, 100 * (1 - contamination_rate))
y_pred = (scores >= threshold).astype(int)

print(f"✓ Anomaly Threshold used: {threshold:.6f}")

cm = confusion_matrix(y, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

print("\n📋 Classification Report:")
print(classification_report(y, y_pred, digits=4, zero_division=0))

# ========================
# VISUALIZATION
# ========================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PR Curve
axes[0].plot(recall, precision, linewidth=2.5, color='#e74c3c')
axes[0].fill_between(recall, precision, alpha=0.3, color='#e74c3c')
axes[0].set_xlabel("Recall", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Precision", fontsize=12, fontweight='bold')
axes[0].set_title(f"Isolation Forest — PR Curve (AUC={pr_auc:.4f})", fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Confusion Matrix
cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im = axes[1].imshow(cm_display, interpolation='nearest', cmap='RdYlGn')
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Benign', 'Malicious'])
axes[1].set_yticklabels(['Benign', 'Malicious'])

# Add text annotations
for i in range(2):
    for j in range(2):
        text = axes[1].text(j, i, f'{cm_display[i, j]:.2%}',
                          ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig('anomaly_evaluation.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to anomaly_evaluation.png")
plt.show()