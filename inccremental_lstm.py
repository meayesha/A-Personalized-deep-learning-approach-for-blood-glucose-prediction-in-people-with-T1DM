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

# -------------------------
# Evaluation functions
# -------------------------
def evaluate_vanilla(model, test_data, history_window, n_features, model_out=None, verbose=0):
    """
    Evaluate vanilla (non-incremental) model: batch predict per-day and compute per-day RMSEs.
    test_data: dict mapping subject_id -> subject_dataframe
    """
    results = {}
    for subj, subj_df in test_data.items():
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

def evaluate_incremental(model, test_data, history_window, n_features, incremental_epochs=3, inc_lr=1e-4, model_out=None, verbose=0, lr_main=1e-3):
    """
    Evaluate incremental model: predict sample-by-sample for each day, compute day RMSE,
    then fine-tune model on that day's true examples.
    The model is updated in-place.
    """
    results = {}
    for subj, subj_df in test_data.items():
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

# -------------------------
# CLI / main
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incremental LSTM (self-contained using your process_data) for univariate CGM")
    parser.add_argument('--data_dir', type=str, default='/mnt/data', help='root data dir (contains OhioT1DM-training/imputed and OhioT1DM-testing/imputed)')
    parser.add_argument('--history', type=int, default=12, help='history window length (timesteps)')
    parser.add_argument('--pred_window', type=int, default=30, help='prediction horizon in minutes')
    parser.add_argument('--initial_epochs', type=int, default=20)
    parser.add_argument('--inc_epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--inc_lr', type=float, default=1e-4)
    parser.add_argument('--normalize', action='store_true', help='if your pickles are named windowed_normalized_30min.pickle')
    parser.add_argument('--models_inc_out', type=str, default='./models_incremental', help='where to save incremental models')
    parser.add_argument('--models_vanilla_out', type=str, default='./models_vanilla', help='where to save vanilla models')
    args = parser.parse_args()

    # set globals used by process_data
    prediction_window = args.pred_window
    # keep original names for process_data usage
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

    # locate pickles produced by preprocess (windowed_*min.pickle)
    ph = str(PRED_WINDOW)
    substring = ('normalized_' if args.normalize else '') + ph + 'min'
    train_pickle = os.path.join(DATA_DIR, 'OhioT1DM-training', 'imputed', f'windowed_{substring}.pickle')
    test_pickle  = os.path.join(DATA_DIR, 'OhioT1DM-testing', 'imputed', f'windowed_{substring}.pickle')

    if not os.path.isfile(train_pickle) or not os.path.isfile(test_pickle):
        raise FileNotFoundError(f"Expected pickles not found. Check paths:\n{train_pickle}\n{test_pickle}")

    with open(train_pickle, 'rb') as f:
        train_data = pickle.load(f, encoding='latin1')
    with open(test_pickle, 'rb') as f:
        test_data = pickle.load(f, encoding='latin1')

    # pooled training dataset
    all_train_X = None; all_train_y = None
    for sid, subj_df in train_data.items():
        X_sub, y_sub, _ = process_data(subj_df)
        if X_sub.shape[0] == 0:
            continue
        if all_train_X is None:
            all_train_X = X_sub; all_train_y = y_sub
        else:
            all_train_X = np.concatenate((all_train_X, X_sub), axis=0)
            all_train_y = np.concatenate((all_train_y, y_sub), axis=0)

    if all_train_X is None or all_train_X.shape[0] == 0:
        raise RuntimeError("No training samples extracted from train pickles. Check preprocess outputs and process_data assumptions.")

    # Ensure shapes for LSTM
    N_FEATURES = 1
    all_train_X = shape_X_for_lstm(all_train_X, HISTORY, N_FEATURES)
    all_train_y = np.array(all_train_y).reshape((-1, 1))

    print("Building base LSTM model and training on pooled training subjects ...")
    base_model = build_model(HISTORY, N_FEATURES, out_dim=1, units=STATE_VECTOR_LENGTH, lr=LR)

    # small validation split from pooled data (5%) if enough samples
    num_train = all_train_X.shape[0]
    val_frac = 0.05
    if num_train > 50:
        split = int(num_train * (1.0 - val_frac))
        X_tr = all_train_X[:split]; y_tr = all_train_y[:split]
        X_val = all_train_X[split:]; y_val = all_train_y[split:]
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        base_model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=INITIAL_EPOCHS, batch_size=BATCH, callbacks=[es], verbose=1)
    else:
        base_model.fit(all_train_X, all_train_y, epochs=INITIAL_EPOCHS, batch_size=BATCH, verbose=1)

    # Experiment 1: Vanilla
    print("\n=== Running vanilla (non-incremental) evaluation ===")
    vanilla_model = build_model(HISTORY, N_FEATURES, out_dim=1, units=STATE_VECTOR_LENGTH, lr=LR)
    vanilla_model.set_weights(base_model.get_weights())
    vanilla_results = evaluate_vanilla(vanilla_model, test_data, HISTORY, N_FEATURES, model_out=args.models_vanilla_out, verbose=0)

    # Experiment 2: Incremental
    print("\n=== Running incremental evaluation (predict -> update per day) ===")
    inc_model = build_model(HISTORY, N_FEATURES, out_dim=1, units=STATE_VECTOR_LENGTH, lr=LR)
    inc_model.set_weights(base_model.get_weights())
    incremental_results = evaluate_incremental(inc_model, test_data, HISTORY, N_FEATURES, incremental_epochs=INC_EPOCHS, inc_lr=INC_LR, model_out=args.models_inc_out, verbose=0, lr_main=LR)

    # Save summary (both experiments)
    summary = {'vanilla': vanilla_results, 'incremental': incremental_results}
    outpath = os.path.join('.', 'incremental_lstm_summary.pickle')
    with open(outpath, 'wb') as f:
        pickle.dump(summary, f)
    print(f"\nSaved summary to {outpath}")

    # Print brief overall numbers
    def overall_mean(results):
        all_vals = []
        for subj_res in results.values():
            all_vals.extend(subj_res['per_day_rmse'])
        return float(np.mean(all_vals)) if all_vals else None

    vmean = overall_mean(vanilla_results)
    imean = overall_mean(incremental_results)
    if vmean is not None:
        print(f"Vanilla overall mean RMSE: {vmean:.3f}")
    if imean is not None:
        print(f"Incremental overall mean RMSE: {imean:.3f}")
