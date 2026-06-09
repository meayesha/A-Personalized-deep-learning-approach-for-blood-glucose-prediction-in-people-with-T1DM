#!/usr/bin/env python3

# coding: utf-8
# ---------------------------------------------------------------
# This code can be used to do blood glucose prediction using 
# OhioT1DM dataset to handle the cold start problem in CGM
# ------------------------------------------------------------------
# Approach:
# ----------------------------------------------------------------------------
# A) Vanilla LSTM (non-incremental) — train once, test on all days without updates.
# B) Incremental LSTM (update window = 1 day) — predict day-by-day, compute day RMSE,
#    then fine-tune model on that day's true examples before moving to next day.
#------------------------------------------------------------------
# Author: Ayesha Parveen
#------------------------------------------------------------------
# References:
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
# https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in.html
# ---------------------------------------------------------------


import os
import argparse
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# -------------------------
# Global defaults (will be set by CLI)
# -------------------------
STATE_VECTOR_LENGTH = 32
ACTIVATION = 'relu'

# these globals are used by process_data (matching your main.py usage)
prediction_window = 30
prediction_type = 'single'
dimension = 'univariate'

# -------------------------
# process_data copied from your main.py
# -------------------------
def process_data(df):
    """
    Exactly the process_data you provided from main.py.
    Expects df to be a pandas DataFrame in the windowed format your preprocess creates.
    Returns: X, y, dates (X,y are numpy arrays; dates is the extracted date-columns array)
    """
    # extract date columns (columns whose name starts with 'date')
    dates = df.filter(regex='^date', axis=1).values  # extract datestamps
    # remove date columns from df
    df = df.loc[:, ~df.columns.str.startswith('date')]

    if prediction_type == 'single':
        # drop intermediate future columns leaving only a single future reading in the final column
        # matches your original pipeline: gets a single reading 30 minutes into the future
        df.drop(df.columns[-prediction_window:-1], axis=1, inplace=True)
    if dimension == 'univariate':
        # if univariate, keep only CGM columns
        df = df.loc[:, df.columns.str.startswith('CGM')]

    data = df.values
    data = data.astype('float32')

    if prediction_type == 'single':
        X, y = data[:, :-1], data[:, -1:]  # x(t+5)
    elif prediction_type == 'multi':
        X, y = data[:, :-prediction_window], data[:, -prediction_window:]  # x(t), x(t+1), ... , x(t+5)

    return X, y, dates

# -------------------------
# Utilities
# -------------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def shape_X_for_lstm(X, history_window, n_features):
    """
    If X already 3D, return. Otherwise assume X is 2D with shape (samples, history_window * n_features)
    and reshape to (samples, history_window, n_features).
    """
    X = np.array(X)
    if X.ndim == 3:
        return X
    # defensive: if second dim isn't divisible by history_window, try to infer features=1
    if X.shape[1] == history_window:
        return X.reshape((X.shape[0], history_window, n_features))
    # if number of columns equals history_window * n_features (typical), reshape directly
    if X.shape[1] % history_window == 0:
        inferred_nf = X.shape[1] // history_window
        return X.reshape((X.shape[0], history_window, inferred_nf))
    # fallback: reshape to (samples, history_window, n_features) by truncation/padding
    needed = history_window * n_features
    if X.shape[1] >= needed:
        X2 = X[:, :needed]
        return X2.reshape((X2.shape[0], history_window, n_features))
    else:
        # pad with zeros if too short
        pad = np.zeros((X.shape[0], needed - X.shape[1]), dtype=X.dtype)
        X2 = np.concatenate([X, pad], axis=1)
        return X2.reshape((X2.shape[0], history_window, n_features))

def build_model(history_window, n_features, out_dim=1, units=STATE_VECTOR_LENGTH, lr=1e-3):
    model = Sequential()
    model.add(LSTM(units, activation=ACTIVATION, return_sequences=False, input_shape=(history_window, n_features)))
    model.add(Dense(out_dim))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mse'])
    return model

def find_windowed_pickle(data_dir, split_name, substring):
    """
    Locate windowed pickle files from either main.py or preprocess_ohio.py layouts.
    """
    candidates = [
        os.path.join(data_dir, split_name, 'imputed', f'windowed_{substring}.pickle'),
        os.path.join(data_dir, 'csv_files', split_name, 'imputed', f'windowed_{substring}.pickle'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]

def load_subject_data(data_dir, pred_window, normalize):
    ph = str(pred_window)
    substring = ('normalized_' if normalize else '') + ph + 'min'
    pickles = {}
    for split in ('OhioT1DM-training', 'OhioT1DM-testing'):
        path = find_windowed_pickle(data_dir, split, substring)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Expected preprocessed pickle not found. Run preprocess_ohio.py first.\n"
                f"Tried: {path}"
            )
        with open(path, 'rb') as f:
            pickles[split] = pickle.load(f, encoding='latin1')

    all_subjects = {}
    all_subjects.update(pickles['OhioT1DM-training'])
    all_subjects.update(pickles['OhioT1DM-testing'])
    return all_subjects

def concat_subject_arrays(subject_data, subject_ids):
    all_X = None
    all_y = None
    for sid in subject_ids:
        X_sub, y_sub, _ = process_data(subject_data[sid])
        if X_sub.shape[0] == 0:
            continue
        if all_X is None:
            all_X = X_sub
            all_y = y_sub
        else:
            all_X = np.concatenate((all_X, X_sub), axis=0)
            all_y = np.concatenate((all_y, y_sub), axis=0)
    return all_X, all_y

def train_base_model(X, y, history_window, n_features, initial_epochs, batch, lr):
    X = shape_X_for_lstm(X, history_window, n_features)
    y = np.array(y).reshape((-1, 1))
    model = build_model(history_window, n_features, out_dim=1, units=STATE_VECTOR_LENGTH, lr=lr)

    num_train = X.shape[0]
    val_frac = 0.05
    if num_train > 50:
        split = int(num_train * (1.0 - val_frac))
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        model.fit(
            X[:split], y[:split],
            validation_data=(X[split:], y[split:]),
            epochs=initial_epochs,
            batch_size=batch,
            callbacks=[es],
            verbose=0,
        )
    else:
        model.fit(X, y, epochs=initial_epochs, batch_size=batch, verbose=0)
    return model

def summarize_subject_results(results):
    rows = []
    for subj, res in sorted(results.items()):
        per_day = res.get('per_day_rmse') or []
        avg = res.get('mean_rmse')
        best = res.get('best_rmse')
        worst = res.get('worst_rmse')
        if avg is None and per_day:
            avg = float(np.mean(per_day))
        if best is None and per_day:
            best = float(np.min(per_day))
        if worst is None and per_day:
            worst = float(np.max(per_day))
        rows.append({
            'Subject': subj,
            'Average': avg,
            'Best': best,
            'Worst': worst,
            'Days': len(per_day),
        })
    return pd.DataFrame(rows)

def print_results_table(title, results):
    df = summarize_subject_results(results)
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if df['Average'].notna().any():
        print(f"Overall mean RMSE: {df['Average'].mean():.3f}")

def plot_per_day_rmse(results, output_dir, experiment_name):
    if not HAS_MPL:
        return
    os.makedirs(output_dir, exist_ok=True)
    for subj, res in sorted(results.items()):
        per_day = res.get('per_day_rmse') or []
        if not per_day:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(per_day) + 1), per_day, marker='o', markersize=3)
        plt.xlabel('Day')
        plt.ylabel('RMSE (mg/dL)')
        plt.title(f'{experiment_name} — Subject P{subj}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{experiment_name}_P{subj}.png'))
        plt.close()

# -------------------------
# Evaluation functions
# -------------------------
def evaluate_vanilla(model, test_data, history_window, n_features, model_out=None, verbose=0, subject_ids=None):
    """
    Evaluate vanilla (non-incremental) model: batch predict per-day and compute per-day RMSEs.
    test_data: dict mapping subject_id -> subject_dataframe
    """
    results = {}
    subjects = subject_ids if subject_ids is not None else test_data.keys()
    for subj in subjects:
        subj_df = test_data[subj]
        X_all, y_all, dates = process_data(subj_df)
        if X_all.shape[0] == 0:
            results[subj] = {'per_day_rmse': [], 'mean_rmse': None, 'days_count': 0}
            continue

        X_all = shape_X_for_lstm(X_all, history_window, n_features)
        y_all = np.array(y_all).reshape((-1, 1))

        # convert dates (array of date columns) to calendar days if possible
        try:
            dates_dt = [pd.to_datetime(d[-1]) if hasattr(d, '__len__') else pd.to_datetime(d) for d in dates]
            days = [dt.date() for dt in dates_dt]
        except Exception:
            # fallback: use integer indices
            days = list(range(X_all.shape[0]))

        unique_days = sorted(list(dict.fromkeys(days)))
        per_day_rmse = []
        for day in unique_days:
            idxs = [i for i, d in enumerate(days) if d == day]
            if not idxs:
                continue
            X_day = X_all[idxs]
            y_day = y_all[idxs].flatten()
            y_hat = model.predict(X_day, verbose=0).flatten()
            day_rmse = rmse(y_day, y_hat)
            per_day_rmse.append(day_rmse)

        results[subj] = {'per_day_rmse': per_day_rmse,
                         'mean_rmse': float(np.mean(per_day_rmse)) if per_day_rmse else None,
                         'days_count': len(per_day_rmse)}

        if model_out:
            os.makedirs(model_out, exist_ok=True)
            model.save(os.path.join(model_out, f"vanilla_{subj}.h5"))

    return results

def evaluate_incremental(model, test_data, history_window, n_features, incremental_epochs=3, inc_lr=1e-4, model_out=None, verbose=0, lr_main=1e-3, subject_ids=None):
    """
    Evaluate incremental model: predict sample-by-sample for each day, compute day RMSE,
    then fine-tune model on that day's true examples.
    The model is updated in-place.
    """
    results = {}
    subjects = subject_ids if subject_ids is not None else test_data.keys()
    for subj in subjects:
        subj_df = test_data[subj]
        print(f"[incremental] processing subject {subj} ...")
        X_all, y_all, dates = process_data(subj_df)
        if X_all.shape[0] == 0:
            results[subj] = {'per_day_rmse': [], 'mean_rmse': None, 'days_count': 0}
            continue

        X_all = shape_X_for_lstm(X_all, history_window, n_features)
        y_all = np.array(y_all).reshape((-1, 1))

        # convert dates
        try:
            dates_dt = [pd.to_datetime(d[-1]) if hasattr(d, '__len__') else pd.to_datetime(d) for d in dates]
            days = [dt.date() for dt in dates_dt]
        except Exception:
            days = list(range(X_all.shape[0]))

        unique_days = sorted(list(dict.fromkeys(days)))
        per_day_rmse = []

        for day in unique_days:
            idxs = [i for i, d in enumerate(days) if d == day]
            if not idxs:
                continue

            preds = []
            trues = []
            # predict sample-by-sample (simulate online)
            for i in idxs:
                x_i = X_all[i:i+1]
                y_true_val = float(y_all[i][0])
                y_hat = model.predict(x_i, verbose=0)
                y_hat_val = float(y_hat.flatten()[0])
                preds.append(y_hat_val)
                trues.append(y_true_val)

            day_rmse = rmse(trues, preds)
            per_day_rmse.append(day_rmse)
            print(f"Subject {subj} | Day {day} | samples {len(idxs)} | day RMSE {day_rmse:.3f}")

            # fine-tune on this day's true examples
            X_day = X_all[idxs]
            y_day = y_all[idxs]
            if X_day.shape[0] > 1:
                # temporarily lower LR for incremental fine-tuning (safer)
                model.compile(optimizer=Adam(learning_rate=inc_lr), loss='mse', metrics=['mse'])
                model.fit(X_day, y_day, epochs=incremental_epochs,
                          batch_size=min(64, max(1, X_day.shape[0]//2)), verbose=0)
                # restore main LR
                model.compile(optimizer=Adam(learning_rate=lr_main), loss='mse', metrics=['mse'])

        results[subj] = {'per_day_rmse': per_day_rmse,
                         'mean_rmse': float(np.mean(per_day_rmse)) if per_day_rmse else None,
                         'days_count': len(per_day_rmse)}

        if model_out:
            os.makedirs(model_out, exist_ok=True)
            model.save(os.path.join(model_out, f"inc_{subj}.h5"))

    return results

def run_vanilla_with_repeats(subject_data, test_subj, history_window, n_features, train_ids,
                           initial_epochs, batch, lr, vanilla_runs, model_out=None):
    """
    Thesis baseline: retrain a fresh model vanilla_runs times on n-1 subjects,
    then evaluate on the held-out subject without any online updates.
    """
    per_run_means = []
    per_run_best = []
    per_run_worst = []
    per_run_days = []

    for run_idx in range(vanilla_runs):
        base_model = train_base_model(
            *concat_subject_arrays(subject_data, train_ids),
            history_window, n_features, initial_epochs, batch, lr,
        )
        vanilla_model = build_model(history_window, n_features, out_dim=1, units=STATE_VECTOR_LENGTH, lr=lr)
        vanilla_model.set_weights(base_model.get_weights())
        run_result = evaluate_vanilla(
            vanilla_model,
            subject_data,
            history_window,
            n_features,
            model_out=model_out if run_idx == vanilla_runs - 1 else None,
            subject_ids=[test_subj],
        )
        res = run_result[test_subj]
        per_day = res['per_day_rmse']
        per_run_means.append(res['mean_rmse'])
        per_run_best.append(float(np.min(per_day)))
        per_run_worst.append(float(np.max(per_day)))
        per_run_days.append(per_day)

    return {
        test_subj: {
            'per_day_rmse': per_run_days[-1],
            'mean_rmse': float(np.mean(per_run_means)),
            'best_rmse': float(np.min(per_run_best)),
            'worst_rmse': float(np.max(per_run_worst)),
            'days_count': len(per_run_days[-1]),
            'run_means': per_run_means,
        }
    }

def run_incremental_for_subject(subject_data, test_subj, history_window, n_features, train_ids,
                                initial_epochs, batch, lr, inc_epochs, inc_lr, model_out=None):
    base_model = train_base_model(
        *concat_subject_arrays(subject_data, train_ids),
        history_window, n_features, initial_epochs, batch, lr,
    )
    inc_model = build_model(history_window, n_features, out_dim=1, units=STATE_VECTOR_LENGTH, lr=lr)
    inc_model.set_weights(base_model.get_weights())
    result = evaluate_incremental(
        inc_model,
        subject_data,
        history_window,
        n_features,
        incremental_epochs=inc_epochs,
        inc_lr=inc_lr,
        model_out=model_out,
        lr_main=lr,
        subject_ids=[test_subj],
    )
    per_day = result[test_subj]['per_day_rmse']
    result[test_subj]['best_rmse'] = float(np.min(per_day)) if per_day else None
    result[test_subj]['worst_rmse'] = float(np.max(per_day)) if per_day else None
    return result

# -------------------------
# CLI / main
# -------------------------
if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(repo_root, 'data')

    parser = argparse.ArgumentParser(
        description="Incremental LSTM for univariate CGM (thesis-aligned leave-one-subject-out evaluation)"
    )
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='root data dir containing OhioT1DM XML/pickles')
    parser.add_argument('--history', type=int, default=12, help='history window length (timesteps)')
    parser.add_argument('--pred_window', type=int, default=30, help='prediction horizon in minutes')
    parser.add_argument('--initial_epochs', type=int, default=20)
    parser.add_argument('--inc_epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--inc_lr', type=float, default=1e-4)
    parser.add_argument('--vanilla_runs', type=int, default=5,
                        help='number of training repeats for vanilla baseline (thesis uses 5)')
    parser.add_argument('--subjects', type=str, default='',
                        help='comma-separated subject IDs to run (default: all 12)')
    parser.add_argument('--pooled', action='store_true',
                        help='use original pooled train/test split instead of leave-one-subject-out')
    parser.add_argument('--normalize', action='store_true',
                        help='if your pickles are named windowed_normalized_30min.pickle')
    parser.add_argument('--plot', action='store_true', help='save per-day RMSE curves as PNG files')
    parser.add_argument('--models_inc_out', type=str, default='./models_incremental')
    parser.add_argument('--models_vanilla_out', type=str, default='./models_vanilla')
    parser.add_argument('--results_dir', type=str, default='./results_incremental')
    args = parser.parse_args()

    prediction_window = args.pred_window
    if prediction_window in (30, 60):
        prediction_window = prediction_window // 5
    prediction_type = 'single'
    dimension = 'univariate'

    DATA_DIR = args.data_dir
    HISTORY = args.history
    PRED_WINDOW = args.pred_window
    INITIAL_EPOCHS = args.initial_epochs
    INC_EPOCHS = args.inc_epochs
    BATCH = args.batch
    LR = args.lr
    INC_LR = args.inc_lr
    N_FEATURES = 1

    subject_data = load_subject_data(DATA_DIR, PRED_WINDOW, args.normalize)
    all_subjects = sorted(subject_data.keys())
    if args.subjects.strip():
        selected = [s.strip() for s in args.subjects.split(',') if s.strip()]
        missing = [s for s in selected if s not in subject_data]
        if missing:
            raise ValueError(f"Unknown subject IDs: {missing}")
        all_subjects = selected

    print(f"Loaded {len(all_subjects)} subjects: {', '.join(all_subjects)}")
    os.makedirs(args.results_dir, exist_ok=True)

    vanilla_results = {}
    incremental_results = {}

    if args.pooled:
        train_ids = [s for s in all_subjects if s in ('559', '563', '570', '575', '588', '591')]
        test_ids = [s for s in all_subjects if s not in train_ids]
        if not train_ids or not test_ids:
            raise RuntimeError("Pooled mode expects standard Ohio split (6 train + 6 test subjects).")

        print("\n=== Pooled mode: train on OhioT1DM-training subjects, test on OhioT1DM-testing subjects ===")
        base_model = train_base_model(
            *concat_subject_arrays(subject_data, train_ids),
            HISTORY, N_FEATURES, INITIAL_EPOCHS, BATCH, LR,
        )
        vanilla_model = build_model(HISTORY, N_FEATURES, out_dim=1, units=STATE_VECTOR_LENGTH, lr=LR)
        vanilla_model.set_weights(base_model.get_weights())
        vanilla_results = evaluate_vanilla(
            vanilla_model, subject_data, HISTORY, N_FEATURES,
            model_out=args.models_vanilla_out, subject_ids=test_ids,
        )
        inc_model = build_model(HISTORY, N_FEATURES, out_dim=1, units=STATE_VECTOR_LENGTH, lr=LR)
        inc_model.set_weights(base_model.get_weights())
        incremental_results = evaluate_incremental(
            inc_model, subject_data, HISTORY, N_FEATURES,
            incremental_epochs=INC_EPOCHS, inc_lr=INC_LR,
            model_out=args.models_inc_out, lr_main=LR, subject_ids=test_ids,
        )
    else:
        print("\n=== Leave-one-subject-out mode (thesis setup) ===")
        for test_subj in all_subjects:
            train_ids = [s for s in all_subjects if s != test_subj]
            print(f"\n--- Held-out subject P{test_subj} | train on {len(train_ids)} subjects ---")

            vanilla_results.update(
                run_vanilla_with_repeats(
                    subject_data, test_subj, HISTORY, N_FEATURES, train_ids,
                    INITIAL_EPOCHS, BATCH, LR, args.vanilla_runs,
                    model_out=args.models_vanilla_out,
                )
            )
            incremental_results.update(
                run_incremental_for_subject(
                    subject_data, test_subj, HISTORY, N_FEATURES, train_ids,
                    INITIAL_EPOCHS, BATCH, LR, INC_EPOCHS, INC_LR,
                    model_out=args.models_inc_out,
                )
            )

    summary = {'vanilla': vanilla_results, 'incremental': incremental_results}
    outpath = os.path.join(args.results_dir, 'incremental_lstm_summary.pickle')
    with open(outpath, 'wb') as f:
        pickle.dump(summary, f)
    print(f"\nSaved summary to {outpath}")

    print_results_table('Table 4.1 style — Vanilla (non-incremental) RMSE by subject', vanilla_results)
    print_results_table('Table 4.2 style — Incremental RMSE by subject', incremental_results)

    vanilla_csv = os.path.join(args.results_dir, 'vanilla_subject_rmse.csv')
    incremental_csv = os.path.join(args.results_dir, 'incremental_subject_rmse.csv')
    summarize_subject_results(vanilla_results).to_csv(vanilla_csv, index=False)
    summarize_subject_results(incremental_results).to_csv(incremental_csv, index=False)
    print(f"Saved CSV summaries to {vanilla_csv} and {incremental_csv}")

    if args.plot:
        plot_dir = os.path.join(args.results_dir, 'plots')
        plot_per_day_rmse(vanilla_results, os.path.join(plot_dir, 'vanilla'), 'Vanilla')
        plot_per_day_rmse(incremental_results, os.path.join(plot_dir, 'incremental'), 'Incremental')
        print(f"Saved per-day RMSE plots under {plot_dir}")
