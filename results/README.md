# Results

Generated training outputs are written here when running:

```powershell
python src/train.py --data data/breast-cancer-wisconsin.data --output-dir results
```

Expected generated files include:

- `model_results.csv`
- `classification_report.json`
- `confusion_matrix.png`
- `feature_importance.png`, when the selected model supports feature importances
