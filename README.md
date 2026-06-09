# Personalized Deep Learning for Blood Glucose Prediction in T1DM

Master's thesis implementation (Stevens Institute of Technology, 2021) for **personalized blood glucose forecasting** on the [OhioT1DM](https://github.com/jxx123/sleepbench) dataset using LSTM models.

The main contribution is an **incremental LSTM** that addresses the **cold-start problem**: a general model is trained on other patients, then fine-tuned day-by-day as new CGM data from the target patient becomes available.

## Repository structure

```
.
├── data/                      # Raw OhioT1DM XML files (see Data setup below)
├── preprocess_ohio.py         # XML → CSV → imputation → windowed pickles
├── preprocess_oaps.py         # Preprocessing for OpenAPS (teacher models)
├── preprocess.sh              # Runs Ohio preprocessing with default settings
├── main.py                    # Teacher/student learning pipelines (Approaches I–IV)
├── incremental_lstm.py        # Incremental + vanilla LSTM evaluation (thesis method)
├── requirements.txt
└── A personalized deep learning approach for glucose prediction in T!DM.pdf
```

## Requirements

- Python 3.9–3.11 recommended (TensorFlow may not support the newest Python releases yet)
- See `requirements.txt` for Python packages

### Install

```bash
cd A-Personalized-deep-learning-approach-for-blood-glucose-prediction-in-people-with-T1DM

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

On Apple Silicon, if `tensorflow` fails to install, try:

```bash
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt --no-deps
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib python-dateutil pytz
```

## Data setup

Preprocessing expects the OhioT1DM XML layout:

```
data/
├── OhioT1DM-training/
│   ├── 559-ws-training.xml
│   ├── 563-ws-training.xml
│   └── ...
└── OhioT1DM-testing/
    ├── 540-ws-testing.xml
    ├── 544-ws-testing.xml
    └── ...
```

If your files are currently under `data/train/` and `data/test/`, create symlinks from the repo root:

```bash
cd data
ln -s train OhioT1DM-training
ln -s test OhioT1DM-testing
cd ..
```

**Training subjects (2018 split):** 559, 563, 570, 575, 588, 591  
**Testing subjects (2020 split):** 540, 544, 552, 567, 584, 596  

All 12 subjects are used in leave-one-subject-out evaluation for the incremental model.

## Preprocessing

Default settings match the thesis:

| Parameter | Value |
|-----------|-------|
| History window | 12 readings (1 hour at 5-min intervals) |
| Prediction horizon | 30 minutes |
| CGM threshold | Remove values ≤ 15 mg/dL |
| Normalization | Off by default |

Run:

```bash
bash preprocess.sh
```

This writes windowed pickles to:

```
data/csv_files/OhioT1DM-training/imputed/windowed_30min.pickle
data/csv_files/OhioT1DM-testing/imputed/windowed_30min.pickle
```

To enable normalization, set `normalize_data=True` in `preprocess.sh` and pass `--normalize` when running `incremental_lstm.py`.

## Incremental LSTM (thesis method)

`incremental_lstm.py` implements two experiments:

1. **Vanilla LSTM** — train once, predict all test days with no online updates (baseline).
2. **Incremental LSTM** — for each day: predict → compute daily RMSE → fine-tune on that day’s ground-truth data → move to the next day.

### Default evaluation (leave-one-subject-out)

Train on 11 subjects, hold out 1, repeat for all 12 subjects. Matches Chapter 3.4 of the thesis.

```bash
python incremental_lstm.py --plot
```

### Quick smoke test (one subject)

```bash
python incremental_lstm.py --subjects 567 --vanilla_runs 1
```

### Pooled train/test split (6 + 6 subjects)

```bash
python incremental_lstm.py --pooled
```

### Useful options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `./data` | Root directory with Ohio XML/pickles |
| `--pred_window` | `30` | Prediction horizon in minutes |
| `--history` | `12` | LSTM input timesteps |
| `--initial_epochs` | `20` | Base model training epochs |
| `--inc_epochs` | `3` | Incremental fine-tuning epochs per day |
| `--vanilla_runs` | `5` | Vanilla baseline repetitions (thesis uses 5) |
| `--plot` | off | Save per-day RMSE curves (Figures 4.1–4.12 style) |
| `--results_dir` | `./results_incremental` | Output directory for metrics and plots |

### Outputs

After a run, check `./results_incremental/`:

- `incremental_lstm_summary.pickle` — full per-subject results
- `vanilla_subject_rmse.csv` — Table 4.1-style metrics
- `incremental_subject_rmse.csv` — Table 4.2-style metrics
- `plots/` — per-day RMSE curves (with `--plot`)

Saved models (optional):

- `./models_vanilla/`
- `./models_incremental/`

## Main pipelines (`main.py`)

`main.py` runs the HAIL Lab teacher/student experiments on OhioT1DM:

| Pipeline | Description |
|----------|-------------|
| `student` | Train LSTM on Ohio data only |
| `teacher` | Pre-trained OAPS model, no retraining |
| `retrain` | Pre-trained OAPS model, fine-tuned on Ohio |
| `teacher_student` | Soft targets from teacher to train student RNN |

Example:

```bash
python main.py \
  <root_directory> \
  <data_directory> \
  <output_directory> \
  <model_directory> \
  12 30 univariate single False LSTM OhioT1DM True student
```

You need pre-trained teacher weights under `model_directory` for teacher-based pipelines. Preprocess OAPS data with `preprocess_oaps.py` first if using those approaches.

## Model details

| Setting | Value |
|---------|-------|
| Architecture | Single LSTM (32 units, ReLU) + Dense(1) |
| Task | Univariate CGM, single-step 30-min forecast |
| Incremental update window | 1 day |
| Metric | RMSE (mg/dL) |

## References

- Thesis: *A Personalized Deep Learning Approach for Blood Glucose Prediction in People with T1DM* — Ayesha Parveen (2021), Stevens Institute of Technology
- Original HAIL Lab code for data preprocessing: Hadia Hameed (2020)
- OhioT1DM dataset

## Citation

If you use this code, please cite the thesis and the OhioT1DM dataset appropriately.
