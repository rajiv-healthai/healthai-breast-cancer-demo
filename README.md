# HealthAI Breast Health Research Demo

A Streamlit learning demo that classifies five cell measurements with a Logistic Regression model.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project files

- `app.py` — page, input form, and result display.
- `model_service.py` — loads the model and scaler and runs predictions.
- `templates.html` — accessible page sections and dashboard copy.
- `assets/styles.css` — responsive dashboard styling.
- `assets/neural_brain.svg` — animated neural brain artwork used in the hero.
- `data/training_data.csv` — 569 labeled records from the UCI Wisconsin Diagnostic Breast Cancer dataset, reduced to the five inputs used by the app.
- `train_model.ipynb` — numbered notebook cells with dataset descriptions, statistical analysis, model training, and evaluation.
- `model.pkl`, `scaler.pkl` — trained app artifacts.

The CSV columns are `sample_id`, `mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`, `mean_concavity`, and `target`. The model uses the five measurements in that order; `sample_id` is for reference only. Target `0` means malignant and `1` means benign.

Open `train_model.ipynb` from the project folder and run its cells in order. It evaluates a stratified 20% holdout, then refits on all rows and saves `model.pkl` and `scaler.pkl`. The save cell replaces those files.

Dataset source: [UCI Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
