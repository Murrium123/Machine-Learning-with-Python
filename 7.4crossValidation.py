from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores=cross_val_score(rf, cancer.data, cancer.target, cv=cv, scoring='f1')

print(f'CV Scores: {scores.round(4)}')
print(f'Mean: {scores.mean():.4f}  (+/-{scores.std()*2:.4f})')