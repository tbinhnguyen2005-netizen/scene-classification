# Generated Model Artifacts

The training pipeline may create PCA, KMeans, Gaussian-mixture, feature-cache, classifier, fusion-weight, and result files in Joblib/Pickle/NumPy formats. They are excluded from Git because they can be large and are environment-dependent.

Regenerate them by running:

```bash
DATA_DIR=/path/to/15_scene python src/scene_classification.py
```

Do not load untrusted Pickle or Joblib files because deserialization can execute arbitrary code.

