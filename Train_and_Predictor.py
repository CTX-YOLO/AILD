import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    roc_auc_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.tree import DecisionTreeClassifier

from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import os
import json
from pathlib import Path
import joblib


# =============================
# Constants Configuration
# =============================

def get_data_path():
    """Get data file path, prioritizing environment variable specified path"""
    default_path = r"../data/clean_data.xlsx"
    return os.environ.get('EXCEL_DATA_PATH', default_path)

KEY_FEATURES = ['length', 'a_M1_dz2', 'a_M3_dxz', 'a_M2_s', 'c_Rs_dz2','num']

TRAIN_SIZE = 0.7
RANDOM_STATE = 45

# Model and artifact directories - support versioning
ARTIFACTS_DIR = Path("artifacts")

def get_model_version():
    """Get model version from environment variable, use default if not available"""
    return os.environ.get('MODEL_VERSION', 'default_weights_files')

def get_versioned_dir(model_type: str):
    """Get versioned model directory"""
    version = get_model_version()
    return ARTIFACTS_DIR / version / model_type

# Default version directories (backward compatibility)
def get_linear_dir(): return get_versioned_dir("linear")
def get_gpr_dir(): return get_versioned_dir("gpr")
def get_dtc_dir(): return get_versioned_dir("dtc")

LINEAR_DIR = get_versioned_dir("linear")
GPR_DIR = get_versioned_dir("gpr")
DTC_DIR = get_versioned_dir("dtc")


# =============================
# Utility Functions
# =============================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def evaluate_with_bootstrap(y_true, y_pred, n_iterations=1000, ci=0.95):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rng = np.random.default_rng(seed=42)
    n = len(y_true)
    
    r2s, rmses, maes = [], [], []
    
    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        
        r2s.append(r2_score(y_true_sample, y_pred_sample))
        rmses.append(np.sqrt(mean_squared_error(y_true_sample, y_pred_sample)))
        maes.append(mean_absolute_error(y_true_sample, y_pred_sample))
    
    def ci_bounds(metric_list):
        lower = np.percentile(metric_list, (1 - ci) / 2 * 100)
        upper = np.percentile(metric_list, (1 + ci) / 2 * 100)
        return np.mean(metric_list), lower, upper
    
    r2_mean, r2_lower, r2_upper = ci_bounds(r2s)
    rmse_mean, rmse_lower, rmse_upper = ci_bounds(rmses)
    mae_mean, mae_lower, mae_upper = ci_bounds(maes)
    
    return {
        "R2": (r2_mean, r2_lower, r2_upper),
        "RMSE": (rmse_mean, rmse_lower, rmse_upper),
        "MAE": (mae_mean, mae_lower, mae_upper),
    }


def metrics_style(metrics_ci):
    for key, (mean_val, lower, upper) in metrics_ci.items():
        error = (upper - lower) / 2
        print(f"{key}: {mean_val:.3f} ± {error:.3f}")





# =============================
# Data Preparation
# =============================
def load_and_scale_regression_data(excel_path=None):
    data_path = excel_path or get_data_path()
    cleaned_data = pd.read_excel(data_path)
    scaler_min_max = MinMaxScaler()
    # Note: Only fit scaler on raw features, but return scaled for model training, keep original DataFrame for saving test set raw values
    data_raw = cleaned_data[KEY_FEATURES].copy()
    data_scaled = scaler_min_max.fit_transform(data_raw)
    data_scaled = pd.DataFrame(data_scaled, columns=KEY_FEATURES, index=cleaned_data.index)
    y = cleaned_data['E_ad']
    stratify_series = cleaned_data['TYPE'] if 'TYPE' in cleaned_data.columns else None
    return data_scaled, y, stratify_series, scaler_min_max, data_raw


def load_classification_data(excel_path=None):
    data_path = excel_path or get_data_path()
    cluster_data = pd.read_excel(data_path)
    X = cluster_data[KEY_FEATURES]
    y = cluster_data["cluster"]
    return X, y


# =============================
# Artifact Persistence/Loading and General Tools
# =============================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_regression_artifacts(save_dir: Path, model, scaler, X_test: pd.DataFrame, y_test: pd.Series, feature_names, metrics: dict = None):
    ensure_dir(save_dir)

    # Save model and scaler
    if save_dir.name == "linear":
        # statsmodels uses built-in save method
        model.save(str(save_dir / "model.pkl"))
    else:
        joblib.dump(model, save_dir / "model.joblib")
    joblib.dump(scaler, save_dir / "scaler.joblib")

    # Save metadata
    meta = {
        "feature_names": list(feature_names),
        "target": "E_ad",
        "scaler": "MinMaxScaler",
        "version": get_model_version(),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    save_json(save_dir / "meta.json", meta)

    # Save metrics as JSON
    if metrics:
        # Ensure all values are JSON serializable Python native types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        json_metrics = convert_to_native(metrics)
        save_json(save_dir / "metrics.json", json_metrics)


def save_classification_artifacts(save_dir: Path, model, X_test: pd.DataFrame, y_test: pd.Series, feature_names, metrics: dict = None):
    ensure_dir(save_dir)
    joblib.dump(model, save_dir / "model.joblib")

    # 保存元数据
    meta = {
        "feature_names": list(feature_names),
        "target": "cluster",
        "scaler": None,
        "version": get_model_version(),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    save_json(save_dir / "meta.json", meta)

    # Save metrics as JSON
    if metrics:
        # Ensure all values are JSON serializable Python native types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        json_metrics = convert_to_native(metrics)
        save_json(save_dir / "metrics.json", json_metrics)


def parse_values_to_dataframe(values: str, feature_names) -> pd.DataFrame:
    # Support comma or space separated 6 numerical values
    if "," in values:
        parts = [p.strip() for p in values.split(",") if p.strip() != ""]
    else:
        parts = [p.strip() for p in values.split() if p.strip() != ""]
    if len(parts) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} feature values, received {len(parts)}")
    nums = [float(x) for x in parts]
    df = pd.DataFrame([nums], columns=feature_names)
    return df


def split_regression_data(excel_path=None):
    X_data, y_data, stratify_series, scaler, X_raw = load_and_scale_regression_data(excel_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data,
        y_data,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE,
        shuffle=True,
        stratify=stratify_series if stratify_series is not None else None,
    )
    # Synchronously get unscaled raw test set (aligned by index)
    X_test_raw = X_raw.loc[X_test.index]
    return X_train, X_test, y_train, y_test, scaler, X_test_raw


# =============================
# Training and Evaluation: Regression
# =============================
def train_and_report_linear(X_train, X_test, y_train, y_test):
    X_train_const = sm.add_constant(X_train)
    linear_model = sm.OLS(y_train, X_train_const).fit()
    print(linear_model.summary())

    y_pred_train = linear_model.predict(sm.add_constant(X_train))
    y_pred_test = linear_model.predict(sm.add_constant(X_test))

    train_r2, train_rmse, train_mae = evaluate(y_train, y_pred_train)
    print("Training set", train_r2, train_rmse, train_mae)

    r2, rmse, mae = evaluate(y_test, y_pred_test)
    print("Test set", r2, rmse, mae)

    # 计算置信区间
    test_metrics_ci = evaluate_with_bootstrap(y_test, y_pred_test)
    train_metrics_ci = evaluate_with_bootstrap(y_train, y_pred_train)

    print("Test set:")
    metrics_style(test_metrics_ci)
    print("Training set:")
    metrics_style(train_metrics_ci)

    equation = "y = "
    equation += f"{linear_model.params[0]:.4f}"
    for i in range(1, len(linear_model.params)):
        equation += f" + {linear_model.params[i]:.3f} * {KEY_FEATURES[i-1]}"
    print(equation)

    # 返回模型和完整指标
    metrics = {
        'test_R2': r2,
        'test_RMSE': rmse,
        'test_MAE': mae,
        'train_R2': train_r2,
        'train_RMSE': train_rmse,
        'train_MAE': train_mae,
        'test_ci': test_metrics_ci,
        'train_ci': train_metrics_ci,
        'equation': equation
    }
    return linear_model, metrics


def train_and_report_gpr(X_train, X_test, y_train, y_test):
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
    model_gpr = GaussianProcessRegressor(normalize_y=True, kernel=kernel)
    model_gpr.fit(X_train, y_train)

    y_pred_train, y_train_std = model_gpr.predict(X_train, return_std=True)
    y_pred_test, y_test_std = model_gpr.predict(X_test, return_std=True)

    train_r2, train_rmse, train_mae = evaluate(y_train, y_pred_train)
    print("Training set", train_r2, train_rmse, train_mae)

    r2, rmse, mae = evaluate(y_test, y_pred_test)
    print("Test set", r2, rmse, mae)

    # 计算置信区间
    test_metrics_ci = evaluate_with_bootstrap(y_test, y_pred_test)
    train_metrics_ci = evaluate_with_bootstrap(y_train, y_pred_train)

    print("Test set:")
    metrics_style(test_metrics_ci)
    print("Training set:")
    metrics_style(train_metrics_ci)

    # 返回模型和完整指标
    metrics = {
        'test_R2': r2,
        'test_RMSE': rmse,
        'test_MAE': mae,
        'train_R2': train_r2,
        'train_RMSE': train_rmse,
        'train_MAE': train_mae,
        'test_ci': test_metrics_ci,
        'train_ci': train_metrics_ci
    }
    return model_gpr, metrics


def plot_Bayesian(re_gp,figsize=(10, 6),name="DTC"):
    # 绘制收敛图
    fig, ax = plt.subplots(figsize=figsize)
    plot_convergence(re_gp, ax=ax)

    # 设置标题
    ax.set_title(f"Bayesian Optimization Convergence of {name}", pad=12)

    # 设置坐标轴标签
    ax.set_xlabel("Number of Calls")
    ax.set_ylabel("Best Score")

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', labelsize=14)
    def format_ticklabels(ax):
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
    format_ticklabels(ax)

    # 保存图片
    plt.tight_layout()
    # 显示图
    plt.show()

# =============================
# Training and Evaluation: Classification (DTC + Bayesian Optimization)
# =============================
param_space = [
    Integer(3, 5, name='max_depth'),
    Integer(5, 20, name='min_samples_split'),
    Integer(5, 20, name='min_samples_leaf'),
    Integer(3, 7, name='max_features'),
]


def build_objective(X_train_c, y_train_c):
    @use_named_args(param_space)
    def objective(**params):
        model = DecisionTreeClassifier(
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
            random_state=42,
        )
        scores = cross_val_score(model, X_train_c, y_train_c, cv=10, scoring='accuracy', n_jobs=-1)
        return -scores.mean()

    return objective


def run_gp_minimize_with_progress(objective_func, n_calls=50):
    with tqdm(total=n_calls, desc="Optimization Progress") as pbar:
        def callback(_res):
            if hasattr(pbar, "update"):
                pbar.update(1)
        
        result = gp_minimize(objective_func, param_space, n_calls=n_calls, random_state=42, callback=callback)

    return result


def train_and_report_dtc(X_train_c, X_test_c, y_train_c, y_test_c, result):
    best_params = {
        'max_depth': result.x[0],
        'min_samples_split': result.x[1],
        'min_samples_leaf': result.x[2],
        'max_features': result.x[3],
        'random_state': 42,
    }

    dtc = DecisionTreeClassifier(**best_params)
    dtc.fit(X_train_c, y_train_c)

    y_pred_trc = dtc.predict(X_train_c)
    train_a = accuracy_score(y_train_c, y_pred_trc)
    print(f"train_accuracy_score: {train_a:.2f}")

    y_pred_tec = dtc.predict(X_test_c)
    test_a = accuracy_score(y_test_c, y_pred_tec)
    print(f"test_accuracy_score: {test_a:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test_c, y_pred_tec))

    num_labels = 7
    y_test_bin = label_binarize(y_test_c, classes=[i for i in range(num_labels)])
    y_pred_bin = dtc.predict_proba(X_test_c)

    assert y_pred_bin.shape[1] == y_test_bin.shape[1], "Number of classes inconsistent"

    roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro', multi_class='ovr')
    print(f"\nAUC (macro-averaged): {roc_auc:.2f}")

    f1 = f1_score(y_test_c, y_pred_tec, average='macro')
    precision = precision_score(y_test_c, y_pred_tec, average='macro')
    recall = recall_score(y_test_c, y_pred_tec, average='macro')

    print(f"F1 Score (macro): {f1:.2f}")
    print(f"Precision (macro): {precision:.2f}")
    print(f"Recall (macro): {recall:.2f}")

    # Calculate confusion matrix
    classes = sorted(list(pd.unique(y_test_c)))
    try:
        cm = confusion_matrix(y_test_c, y_pred_tec, labels=classes).tolist()
    except Exception:
        cm = None

    # Calculate training set metrics
    y_pred_trc = dtc.predict(X_train_c)
    train_a = accuracy_score(y_train_c, y_pred_trc)
    train_f1 = f1_score(y_train_c, y_pred_trc, average='macro')
    train_precision = precision_score(y_train_c, y_pred_trc, average='macro')
    train_recall = recall_score(y_train_c, y_pred_trc, average='macro')

    # 返回模型和完整指标
    metrics = {
        'test_Accuracy': test_a,
        'test_F1': f1,
        'test_Precision': precision,
        'test_Recall': recall,
        'test_AUC': roc_auc,
        'train_Accuracy': train_a,
        'train_F1': train_f1,
        'train_Precision': train_precision,
        'train_Recall': train_recall,
        'CM': cm,
        'classes': classes
    }
    return dtc, metrics


# =============================
# Prediction and Test Preview Interface
# =============================

def predict_linear(values: str):
    meta = load_json(LINEAR_DIR / "meta.json")
    feature_names = meta["feature_names"]
    scaler = joblib.load(LINEAR_DIR / "scaler.joblib")
    lin_model = sm.load(str(LINEAR_DIR / "model.pkl"))

    X_df = parse_values_to_dataframe(values, feature_names)
    X_scaled = pd.DataFrame(scaler.transform(X_df), columns=feature_names)
    y_pred = lin_model.predict(sm.add_constant(X_scaled))
    return float(y_pred.iloc[0])


def predict_gpr(values: str):
    meta = load_json(GPR_DIR / "meta.json")
    feature_names = meta["feature_names"]
    scaler = joblib.load(GPR_DIR / "scaler.joblib")
    gpr_model: GaussianProcessRegressor = joblib.load(GPR_DIR / "model.joblib")

    X_df = parse_values_to_dataframe(values, feature_names)
    X_scaled = pd.DataFrame(scaler.transform(X_df), columns=feature_names)
    y_pred = gpr_model.predict(X_scaled)
    return float(y_pred[0])


def predict_dtc(values: str):
    meta = load_json(DTC_DIR / "meta.json")
    feature_names = meta["feature_names"]
    dtc_model: DecisionTreeClassifier = joblib.load(DTC_DIR / "model.joblib")

    X_df = parse_values_to_dataframe(values, feature_names)
    proba = dtc_model.predict_proba(X_df)
    pred = dtc_model.predict(X_df)
    return int(pred[0]), proba[0].tolist()


def preview_test_linear():
    meta = load_json(LINEAR_DIR / "meta.json")
    feature_names = meta["feature_names"]
    scaler = joblib.load(LINEAR_DIR / "scaler.joblib")
    lin_model = sm.load(str(LINEAR_DIR / "model.pkl"))

    X_test = pd.read_csv(LINEAR_DIR / "test_X.csv")
    y_test = pd.read_csv(LINEAR_DIR / "test_y.csv")["E_ad"]

    X_scaled = pd.DataFrame(scaler.transform(X_test[feature_names]), columns=feature_names)
    y_pred = lin_model.predict(sm.add_constant(X_scaled))
    r2, rmse, mae = evaluate(y_test, y_pred)
    print("[Preview-Linear] 测试集:", r2, rmse, mae)


def preview_test_gpr():
    meta = load_json(GPR_DIR / "meta.json")
    feature_names = meta["feature_names"]
    scaler = joblib.load(GPR_DIR / "scaler.joblib")
    gpr_model: GaussianProcessRegressor = joblib.load(GPR_DIR / "model.joblib")

    X_test = pd.read_csv(GPR_DIR / "test_X.csv")
    y_test = pd.read_csv(GPR_DIR / "test_y.csv")["E_ad"]

    X_scaled = pd.DataFrame(scaler.transform(X_test[feature_names]), columns=feature_names)
    y_pred = gpr_model.predict(X_scaled)
    r2, rmse, mae = evaluate(y_test, y_pred)
    print("[Preview-GPR] 测试集:", r2, rmse, mae)


def preview_test_dtc():
    meta = load_json(DTC_DIR / "meta.json")
    feature_names = meta["feature_names"]
    dtc_model: DecisionTreeClassifier = joblib.load(DTC_DIR / "model.joblib")

    X_test = pd.read_csv(DTC_DIR / "test_X.csv")
    y_test = pd.read_csv(DTC_DIR / "test_y.csv")["cluster"]

    y_pred = dtc_model.predict(X_test[feature_names])
    test_a = accuracy_score(y_test, y_pred)
    print(f"[Preview-DTC] test_accuracy_score: {test_a:.2f}")


# =============================
# 主流程
# =============================
def main():
    parser = argparse.ArgumentParser(description="Run independent ML tasks: linear, gpr, dtc, or all; also support predict/preview")
    parser.add_argument(
        "-t", "--task",
        choices=["linear", "gpr", "dtc", "all"],
        help="Which task to run",
        default=None,
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["train", "predict", "preview"],
        help="Mode: train to fit and save; predict to load and predict; preview to evaluate saved test set",
        default="train",
    )
    parser.add_argument(
        "-v", "--values",
        type=str,
        help="6 feature values for prediction, separated by comma or space, order matches KEY_FEATURES",
        default=None,
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        help="Path to Excel file for training (overrides environment variable)",
        default=None,
    )
    args = parser.parse_args()

    # If Excel path provided via command line, set environment variable
    if args.excel_path:
        os.environ['EXCEL_DATA_PATH'] = args.excel_path

    if args.task is None:
        print("Please select a model via --task: linear / gpr / dtc / all")
        print("Example: python Train_and_Predictor.py --task linear --mode train")
        return

    # Training mode: Fit and save models + test sets
    if args.mode == "train":
        # Prepare regression data as needed
        if args.task in ("linear", "gpr", "all"):
            X_train, X_test, y_train, y_test, scaler, X_test_raw = split_regression_data()

        if args.task in ("linear", "all"):
            lin_model, lin_metrics = train_and_report_linear(X_train, X_test, y_train, y_test)
            # Save unscaled test set and metrics
            save_regression_artifacts(LINEAR_DIR, lin_model, scaler, X_test_raw[KEY_FEATURES], y_test, KEY_FEATURES, lin_metrics)

        if args.task in ("gpr", "all"):
            gpr_model, gpr_metrics = train_and_report_gpr(X_train, X_test, y_train, y_test)
            # Save unscaled test set and metrics
            save_regression_artifacts(GPR_DIR, gpr_model, scaler, X_test_raw[KEY_FEATURES], y_test, KEY_FEATURES, gpr_metrics)

        if args.task in ("dtc", "all"):
            X_c, y_c = load_classification_data()
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                X_c, y_c, random_state=RANDOM_STATE, train_size=TRAIN_SIZE, shuffle=True
            )

            objective = build_objective(X_train_c, y_train_c)
            result = run_gp_minimize_with_progress(objective, n_calls=50)

            print("Best Parameters: ", result.x)
            print("Best Score: ", -result.fun)

            plot_Bayesian(result, figsize=(8, 6), name="DTC_Optimization")

            dtc_model, dtc_metrics = train_and_report_dtc(X_train_c, X_test_c, y_train_c, y_test_c, result)
            # Classification does not use scaler
            save_classification_artifacts(DTC_DIR, dtc_model, X_test_c[KEY_FEATURES] if all(f in X_test_c.columns for f in KEY_FEATURES) else X_test_c, y_test_c, KEY_FEATURES if all(f in X_test_c.columns for f in KEY_FEATURES) else list(X_train_c.columns), dtc_metrics)

    # Prediction mode: Load saved model and predict on input values
    elif args.mode == "predict":
        if args.values is None:
            print("predict mode requires 6 feature values via --values, order:")
            print(", ".join(KEY_FEATURES))
            return
        if args.task in ("linear", "all"):
            y = predict_linear(args.values)
            print(f"[Predict-Linear] y_pred = {y:.6f}")
        if args.task in ("gpr", "all"):
            y = predict_gpr(args.values)
            print(f"[Predict-GPR] y_pred = {y:.6f}")
        if args.task in ("dtc", "all"):
            label, proba = predict_dtc(args.values)
            print(f"[Predict-DTC] label = {label}, proba = {proba}")

    # Preview mode: Evaluate saved model on saved test set (should match training)
    elif args.mode == "preview":
        if args.task in ("linear", "all"):
            preview_test_linear()
        if args.task in ("gpr", "all"):
            preview_test_gpr()
        if args.task in ("dtc", "all"):
            preview_test_dtc()
    else:
        print("Unknown mode")


if __name__ == "__main__":
    main()
