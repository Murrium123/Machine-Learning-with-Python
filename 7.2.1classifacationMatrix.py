from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

print(f"\nClassification Metrics\n")

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

rf=RandomForestClassifier(
    n_estimators=100, random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Part 2
print(f"\n\n")

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_proba = rf.predict_proba(X_test) [:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test,y_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0,1],[0,1],'k--',label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positiive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Part 3
print(f"\n7.3 Regression Metrics\n")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_regression(y_true, y_pred, name='Model'):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'{name}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}')


# Part 4
print(f"\n7.4 Cross-Validation\n")

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores=cross_val_score(rf, cancer.data, cancer.target, cv=cv, scoring='f1')

print(f'CV Scores: {scores.round(4)}')
print(f'Mean: {scores.mean():.4f}  (+/-{scores.std()*2:.4f})')

# Part 5
print(f"\n7.5 Bias-Variance Tradeoff\n")

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores=learning_curve(
    rf, cancer.data, cancer.target, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), 
    scoring='f1'
)

plt.figure(figsize=(10,5))
plt.plot(
    train_sizes, train_scores.mean(axis=1),
    label='Training'
)
plt.plot(
    train_sizes,
    val_scores.mean(axis=1),
    label='Validation'
)

plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title('Learning Curves')
plt.legend()
plt.show()
