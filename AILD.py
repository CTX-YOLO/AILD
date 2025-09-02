from doctest import Example
import sys
import random
from typing import List, Dict

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QStackedWidget,
    QComboBox,
    QLineEdit,
    QGridLayout,
    QPlainTextEdit,
    QGroupBox,
    QMessageBox,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QScrollArea,
    QGraphicsView,
    QGraphicsScene,
    QFileDialog,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

import subprocess
import time
import re
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
try:
    import statsmodels.api as sm  # optional at startup; functions re-import if needed
except Exception:
    sm = None
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

KEY_FEATURES: List[str] = ['length', 'a_M1_dz2', 'num', 'a_M2_s', 'a_M3_dxz', 'c_Rs_dz2']
# Graphviz 安装路径
graphviz_bin = r"C:\Program Files\Graphviz\bin"

# 临时添加到 PATH
os.environ["PATH"] += os.pathsep + graphviz_bin

Example_VALUE = {
    "length":2.75971428571428,
    "a_M1_dz2":-1.35660134874695,
    "num":7.0,
    "a_M2_s":3.18566048408146,
    "a_M3_dxz":1.58622602669376,
    "c_Rs_dz2":0.206803972932347,
}
# Model artifacts root with version management
ARTIFACTS_DIR = Path('artifacts')
DEFAULT_VERSION = 'default_weights_files'  # Default version directory name created on first training

def get_version_dir(version: str = DEFAULT_VERSION) -> Path:
    """Get the artifacts directory for a specific version."""
    return ARTIFACTS_DIR / version

def get_linear_dir(version: str = DEFAULT_VERSION) -> Path:
    return get_version_dir(version) / 'linear'

def get_gpr_dir(version: str = DEFAULT_VERSION) -> Path:
    return get_version_dir(version) / 'gpr'

def get_dtc_dir(version: str = DEFAULT_VERSION) -> Path:
    return get_version_dir(version) / 'dtc'

# Default directories for backward compatibility
LINEAR_DIR = get_linear_dir()
GPR_DIR = get_gpr_dir()
DTC_DIR = get_dtc_dir()

# DTC class color mapping
DTC_CLASS_COLOR_DICT = {
    0: '#cae2ee',
    1: '#d1ecb9',
    2: '#fdc2c2',
    3: '#fed9a9',
    4: '#dfd1e6',
    5: '#ffffc2',
    6: '#d09b7e',
}
DTC_CLASS_COLOR_DICT_STR = {str(k): v for k, v in DTC_CLASS_COLOR_DICT.items()}


def _load_example_values() -> Dict[str, float]:
    """Initialize example values for prediction inputs.
    Use Example_VALUE as default values.
    """
    return Example_VALUE.copy()


def _legacy_dir_for(model: str) -> Path:
    return ARTIFACTS_DIR / model


def _exists_linear(version: str = DEFAULT_VERSION) -> bool:
    linear_dir = get_linear_dir(version)
    legacy_dir = _legacy_dir_for('linear')
    new_exists = (linear_dir / 'model.pkl').exists() and (linear_dir / 'scaler.joblib').exists()
    legacy_exists = (legacy_dir / 'model.pkl').exists() and (legacy_dir / 'scaler.joblib').exists()
    return new_exists or legacy_exists


def _exists_gpr(version: str = DEFAULT_VERSION) -> bool:
    gpr_dir = get_gpr_dir(version)
    legacy_dir = _legacy_dir_for('gpr')
    new_exists = (gpr_dir / 'model.joblib').exists() and (gpr_dir / 'scaler.joblib').exists()
    legacy_exists = (legacy_dir / 'model.joblib').exists() and (legacy_dir / 'scaler.joblib').exists()
    return new_exists or legacy_exists


def _exists_dtc(version: str = DEFAULT_VERSION) -> bool:
    dtc_dir = get_dtc_dir(version)
    legacy_dir = _legacy_dir_for('dtc')
    return (dtc_dir / 'model.joblib').exists() or (legacy_dir / 'model.joblib').exists()


def get_available_versions() -> List[str]:
    """Get list of available model versions."""
    if not ARTIFACTS_DIR.exists():
        return []
    versions = []
    for item in ARTIFACTS_DIR.iterdir():
        if item.is_dir():
            # Check if this version has any model files
            has_models = (_exists_linear(item.name) or
                         _exists_gpr(item.name) or
                         _exists_dtc(item.name))
            if has_models:
                versions.append(item.name)
    return sorted(set(versions)) if versions else []


def _compute_linear_metrics(version: str = DEFAULT_VERSION) -> Dict[str, float]:
    """Load metrics from JSON file, or return empty dict if not available."""
    linear_dir = get_linear_dir(version)
    metrics_path = linear_dir / 'metrics.json'
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text(encoding='utf-8'))
        except Exception:
            pass
        return {}


def _compute_gpr_metrics(version: str = DEFAULT_VERSION) -> Dict[str, float]:
    """Load metrics from JSON file, or return empty dict if not available."""
    gpr_dir = get_gpr_dir(version)
    metrics_path = gpr_dir / 'metrics.json'
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text(encoding='utf-8'))
        except Exception:
            pass
        return {}


def _compute_dtc_metrics(version: str = DEFAULT_VERSION) -> Dict[str, float]:
    """Load metrics from JSON file, or return skeleton metrics if not available."""
    dtc_dir = get_dtc_dir(version)
    metrics_path = dtc_dir / 'metrics.json'
    model_path = dtc_dir / 'model.joblib'

    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            print(f"DEBUG: Loaded DTC metrics from {metrics_path}: {list(metrics.keys())}")
            return metrics
        except Exception as e:
            print(f"DEBUG: Failed to load DTC metrics: {e}")

    # If no metrics file but model exists, return skeleton
    if model_path.exists():
        import joblib
        model = joblib.load(model_path)
        classes = list(getattr(model, 'classes_', []))
        # Convert numpy types to Python types for JSON serialization
        skeleton_metrics = {
            'Accuracy': None,
            'F1': None,
            'Precision': None,
            'Recall': None,
            'CM': None,
            'classes': [int(cls) for cls in classes],
            'model_path': str(model_path),
        }
        print(f"DEBUG: No metrics file, returning skeleton: {list(skeleton_metrics.keys())}")
        return skeleton_metrics

    print("DEBUG: No DTC model or metrics found")
    return {}


class TrainWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str, int)  # model, percent
    metrics_ready = QtCore.pyqtSignal(str, dict)
    log = QtCore.pyqtSignal(str)
    finished_all = QtCore.pyqtSignal()

    def __init__(self, parent=None, retrain: bool = True):
        super().__init__(parent)
        self.retrain = retrain
        self.failed: bool = False

    def _run_cmd(self, args: List[str]):
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        percent = 0
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                self.log.emit(line.rstrip())
            # Simple progress advancement to avoid blocking UI
            percent = min(95, percent + 1)
            yield percent
        proc.wait()
        if proc.returncode and proc.returncode != 0:
            self.failed = True
            self.log.emit(f"Process failed with code {proc.returncode}: {' '.join(args)}")

    def _train_one(self, model_key: str):
        # Training
        if self.retrain or not {
            'linear': _exists_linear,
            'gpr': _exists_gpr,
            'dtc': _exists_dtc,
        }[model_key]():
            self.log.emit(f"Training {model_key}...")
            for p in self._run_cmd([sys.executable, 'Train_and_Predictor.py', '--task', model_key, '--mode', 'train']):
                self.progress.emit(model_key, p)
        # Calculate metrics
        self.progress.emit(model_key, 97)
        metrics = {}
        try:
            if model_key == 'linear':
                metrics = _compute_linear_metrics()
            elif model_key == 'gpr':
                metrics = _compute_gpr_metrics()
            else:
                metrics = _compute_dtc_metrics()
        except Exception as e:
            self.log.emit(f"Failed to compute metrics for {model_key}: {e}")
            self.failed = True
        self.metrics_ready.emit(model_key, metrics)
        self.progress.emit(model_key, 100)

    def run(self):
        # DTC retraining is disabled per requirement; only train linear & gpr
        for model_key in ['linear', 'gpr']:
            try:
                self._train_one(model_key)
            except Exception as e:
                self.log.emit(f"Training error ({model_key}): {e}")
        self.finished_all.emit()


class ExcelTrainWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str, int)
    metrics_ready = QtCore.pyqtSignal(str, dict)
    log = QtCore.pyqtSignal(str)
    finished_all = QtCore.pyqtSignal()

    def __init__(self, excel_path: str, version: str = None, parent=None):
        super().__init__(parent)
        self.excel_path = excel_path
        # If no version specified and artifacts directory doesn't exist, use default version name
        if version is None:
            if not ARTIFACTS_DIR.exists():
                self.version = DEFAULT_VERSION
            else:
                self.version = f"user_{int(time.time())}"
        else:
            self.version = version

    def _run_cmd(self, args: List[str]):
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        percent = 0
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                self.log.emit(line.rstrip())
            # Simple progress advancement to avoid blocking UI
            percent = min(95, percent + 1)
            yield percent
        proc.wait()
        if proc.returncode and proc.returncode != 0:
            self.log.emit(f"Training subprocess failed with code {proc.returncode}")
            return False
        return True

    def run(self):
        try:
            self.log.emit(f"Loading Excel: {self.excel_path} for version '{self.version}'")

            # Check if Excel file has cluster column
            df = pd.read_excel(self.excel_path)
            has_cluster = 'cluster' in df.columns
            self.log.emit(f"Excel file has cluster column: {has_cluster}")

            # 设置环境变量来传递版本信息和数据路径
            import os
            os.environ['MODEL_VERSION'] = self.version
            os.environ['EXCEL_DATA_PATH'] = self.excel_path

            # 训练Linear和GPR模型（这两个总是要训练的）
            self.log.emit("Training Linear model...")
            self.progress.emit('linear', 10)
            for p in self._run_cmd([sys.executable, 'Train_and_Predictor.py', '--task', 'linear', '--mode', 'train', '--excel-path', self.excel_path]):
                self.progress.emit('linear', p)
            self.progress.emit('linear', 100)

            # 加载并发送Linear指标
            try:
                linear_metrics = _compute_linear_metrics(self.version)
                self.metrics_ready.emit('linear', linear_metrics)
                self.log.emit("Linear model trained and metrics loaded")
            except Exception as e:
                self.log.emit(f"Failed to load Linear metrics: {e}")
                self.metrics_ready.emit('linear', {})

            self.log.emit("Training GPR model...")
            self.progress.emit('gpr', 10)
            for p in self._run_cmd([sys.executable, 'Train_and_Predictor.py', '--task', 'gpr', '--mode', 'train', '--excel-path', self.excel_path]):
                self.progress.emit('gpr', p)
            self.progress.emit('gpr', 100)

            # 加载并发送GPR指标
            try:
                gpr_metrics = _compute_gpr_metrics(self.version)
                self.metrics_ready.emit('gpr', gpr_metrics)
                self.log.emit("GPR model trained and metrics loaded")
            except Exception as e:
                self.log.emit(f"Failed to load GPR metrics: {e}")
                self.metrics_ready.emit('gpr', {})

            # 处理DTC模型
            if has_cluster:
                self.log.emit("Cluster column detected - training DTC model...")
                self.progress.emit('dtc', 10)
                for p in self._run_cmd([sys.executable, 'Train_and_Predictor.py', '--task', 'dtc', '--mode', 'train', '--excel-path', self.excel_path]):
                    self.progress.emit('dtc', p)
                self.progress.emit('dtc', 100)

                # 加载并发送DTC指标
                try:
                    dtc_metrics = _compute_dtc_metrics(self.version)
                    self.metrics_ready.emit('dtc', dtc_metrics)
                    self.log.emit("DTC model trained and metrics loaded")
                except Exception as e:
                    self.log.emit(f"Failed to load DTC metrics: {e}")
                    self.metrics_ready.emit('dtc', {})
            else:
                # 用户数据中没有cluster字段，不训练DTC模型
                # 尝试加载默认权重文件中的DTC模型用于展示
                self.log.emit("No cluster column in uploaded data - skipping DTC training")
                self.log.emit("Attempting to load DTC model from default weights for display...")

                default_dtc_dir = get_dtc_dir(DEFAULT_VERSION)
                if default_dtc_dir.exists() and (default_dtc_dir / 'model.joblib').exists():
                    try:
                        # 复制默认DTC模型到当前版本目录，用于展示
                        current_dtc_dir = get_dtc_dir(self.version)
                        current_dtc_dir.mkdir(parents=True, exist_ok=True)

                        # 复制模型文件
                        import shutil
                        shutil.copy2(default_dtc_dir / 'model.joblib', current_dtc_dir / 'model.joblib')

                        # 复制meta文件（如果存在）
                        if (default_dtc_dir / 'meta.json').exists():
                            shutil.copy2(default_dtc_dir / 'meta.json', current_dtc_dir / 'meta.json')

                        # 复制metrics文件（如果存在）
                        if (default_dtc_dir / 'metrics.json').exists():
                            shutil.copy2(default_dtc_dir / 'metrics.json', current_dtc_dir / 'metrics.json')

                        # 加载并发送DTC指标
                        dtc_metrics = _compute_dtc_metrics(self.version)
                        self.metrics_ready.emit('dtc', dtc_metrics)
                        self.log.emit(f"Loaded DTC model from default weights for display (version: {self.version})")
                    except Exception as e:
                        self.log.emit(f"Failed to copy default DTC model: {e}")
                        self.metrics_ready.emit('dtc', {})
                else:
                    self.log.emit("No default DTC model available - DTC will not be displayed")
                    self.metrics_ready.emit('dtc', {})

                self.progress.emit('dtc', 100)

        except Exception as e:
            self.log.emit(f"Excel training failed: {e}")
        finally:
            self.finished_all.emit()


class LogConsole(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("background-color: #1e293b; color: #e2e8f0; font-family: 'Courier New'; font-size: 12px;")

    def log(self, text: str, level: str = 'info'):
        color_map = {
            'info': '#3498db',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'error': '#e74c3c',
        }
        color = color_map.get(level, '#3498db')
        self.appendHtml(f"<span style='color:{color}'>[{level.upper()}] {text}</span>")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class MetricsDashboardCanvas(FigureCanvas):
    """Three subplots in one figure: left (Linear metrics), middle (GPR metrics), right (DTC metrics or decision tree)."""

    def __init__(self, parent=None):
        fig = Figure(figsize=(14, 3.6))
        self.ax_left = fig.add_subplot(1, 3, 1)
        self.ax_mid = fig.add_subplot(1, 3, 2)
        self.ax_right = fig.add_subplot(1, 3, 3)
        super().__init__(fig)
        self.setParent(parent)

        self.reg_order = ["R2", "RMSE", "MAE"]
        # keep keys for lookup and labels for display to avoid overlap
        self.cls_order_keys = ["Accuracy", "F1", "Precision", "Recall"]
        self.cls_order_labels = ["Acc.", "F1", "Prec.", "Rec."]
        self.linear = None
        self.gpr = None
        self.dtc = None
        self.tree_axes = None
        self._last_tree_png = None
        self.redraw()

    def update_regression(self, linear: dict = None, gpr: dict = None):
        self.linear = linear
        self.gpr = gpr
        self.redraw()
        self.draw()
        # Force update the canvas display
        self.update()

    def update_classification(self, dtc: dict = None, tree_png: np.ndarray = None):
        self.dtc = dtc
        self._last_tree_png = tree_png
        self.redraw()
        self.draw()
        # Force update the canvas display
        self.update()

    def _safe_vals(self, vals):
        def safe_convert(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.0
            try:
                if isinstance(v, str):
                    # Handle string values that might contain commas or other formatting
                    v = v.replace(',', '')
                return float(v)
            except (ValueError, TypeError):
                return 0.0
        return [safe_convert(v) for v in vals]

    def _safe_float(self, v):
        """Safe float conversion for single values"""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        try:
            if isinstance(v, str):
                v = v.replace(',', '')
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    def redraw(self):
        # Left: Linear & GPR metrics (grouped bars)
        self.ax_left.clear()
        self.ax_left.set_title('Linear & GPR Metrics', fontweight='bold')
        x = np.arange(len(self.reg_order))
        width = 0.35
        any_left = False
        if self.linear is not None:
            # Use test set metrics for visualization if available
            vals_l = self._safe_vals([
                self.linear.get(f'test_{k}', self.linear.get(k)) for k in self.reg_order
            ])
            self.ax_left.bar(x - width/2, vals_l, width, label='Linear', color='#3498db')
            for xi, v in zip(x - width/2, vals_l):
                if v > 0:  # Only show text for non-zero values
                    self.ax_left.text(xi, v, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
            any_left = True
        if self.gpr is not None:
            # Use test set metrics for visualization if available
            vals_g = self._safe_vals([
                self.gpr.get(f'test_{k}', self.gpr.get(k)) for k in self.reg_order
            ])
            self.ax_left.bar(x + width/2, vals_g, width, label='GPR', color='#2ecc71')
            for xi, v in zip(x + width/2, vals_g):
                if v > 0:  # Only show text for non-zero values
                    self.ax_left.text(xi, v, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
            any_left = True
        self.ax_left.set_xticks(x)
        self.ax_left.set_xticklabels(self.reg_order)
        self.ax_left.grid(True, axis='y', alpha=0.2)
        if any_left:
            self.ax_left.legend()
        vals_all = []
        if self.linear is not None:
            vals_all += [self.linear.get(k) for k in self.reg_order]
        if self.gpr is not None:
            vals_all += [self.gpr.get(k) for k in self.reg_order]
        vals_all = [v for v in vals_all if v is not None]
        ymax = max(1.0, max(vals_all) * 1.2) if vals_all else 1.0
        self.ax_left.set_ylim(0, ymax)

        # Middle: DTC metrics bars
        self.ax_mid.clear()
        self.ax_mid.set_title('DTC Metrics', fontweight='bold')
        x2 = np.arange(len(self.cls_order_keys))
        if self.dtc:
            # Use test set metrics for visualization if available
            cls_vals = self._safe_vals([
                self.dtc.get(f'test_{k}', self.dtc.get(k)) for k in self.cls_order_keys
            ])
            # If all values are 0 or None, show a message instead
            if all(v <= 0 for v in cls_vals):
                self.ax_mid.text(0.5, 0.5, 'DTC Model Available\nNo Metrics Yet\n(Train with cluster data)',
                               ha='center', va='center', transform=self.ax_mid.transAxes,
                               fontsize=10, color='#666')
                cls_vals = [0.0] * len(self.cls_order_keys)
        else:
            cls_vals = [0.0] * len(self.cls_order_keys)

        self.ax_mid.bar(x2, cls_vals, color='#9b59b6')
        for xi, v in zip(x2, cls_vals):
            if v > 0:  # Only show text for non-zero values
                self.ax_mid.text(xi, v, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
        self.ax_mid.set_xticks(x2)
        self.ax_mid.set_xticklabels(self.cls_order_labels)
        self.ax_mid.grid(True, axis='y', alpha=0.2)
        ymax2 = max(1.0, max(cls_vals) * 1.2 if cls_vals else 1.0)
        self.ax_mid.set_ylim(0, ymax2)

        # Right: DTC decision tree image (if available), else message
        self.ax_right.clear()
        if self._last_tree_png is not None:
            self.ax_right.imshow(self._last_tree_png)
            self.ax_right.axis('off')
            self.ax_right.set_title('DTC Decision Tree', fontweight='bold')
        else:
            self.ax_right.set_title('DTC Decision Tree (graphviz not available)', fontweight='bold')
            self.ax_right.axis('off')

        self.draw()
        # Force update the canvas display
        self.update()


class TrainingSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.train_button = QPushButton('Start Training')
        self.train_button.setCursor(Qt.PointingHandCursor)
        self.train_button.setStyleSheet("background:#27ae60;color:white;font-weight:600;padding:10px 20px;border-radius:6px;")
        self.train_button.clicked.connect(self.start_training)

        # Progress bars for 3 models
        self.linear_bar = QProgressBar()
        self.linear_bar.setFormat('Linear: %p%')
        self.gpr_bar = QProgressBar()
        self.gpr_bar.setFormat('GPR: %p%')
        self.dtc_bar = QProgressBar()
        self.dtc_bar.setFormat('DTC: %p%')

        # Metrics labels
        self.linear_metrics_lbl = QLabel('Linear => R2: --  RMSE: --  MAE: --')
        self.gpr_metrics_lbl = QLabel('GPR => R2: --  RMSE: --  MAE: --')
        self.dtc_metrics_lbl = QLabel('DTC => Acc: --  F1: --  Prec: --  Rec: --')

        # Stats
        self.acc_value = QLabel('--')
        self.loss_value = QLabel('--')
        self.epoch_value = QLabel('--')
        self.step_value = QLabel('--')
        for lbl in (self.acc_value, self.loss_value, self.epoch_value, self.step_value):
            lbl.setStyleSheet('font-size:14px;font-weight:700;color:#1a2b4c;')

        self.logs = LogConsole(self)

        # Layouts
        # Data source selector (plaintext only here)
        data_source_box = QGroupBox('Data source')
        ds_layout = QHBoxLayout()
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(['From Weights', 'From Excel'])
        self.select_data_btn = QPushButton('Select Excel')
        self.select_data_btn.setEnabled(False)
        self.selected_excel_path = None
        def on_ds_change(text):
            self.select_data_btn.setEnabled(text == 'From Excel')
        self.data_source_combo.currentTextChanged.connect(on_ds_change)
        def pick_excel():
            path, _ = QFileDialog.getOpenFileName(self, 'Select Excel file', '.', 'Excel Files (*.xlsx *.xls)')
            if path:
                self.selected_excel_path = path
                self.logs.log(f"Selected Excel: {path}")
        self.select_data_btn.clicked.connect(pick_excel)
        ds_layout.addWidget(QLabel('Use'))
        ds_layout.addWidget(self.data_source_combo)
        ds_layout.addWidget(self.select_data_btn)
        ds_layout.addStretch(1)
        data_source_box.setLayout(ds_layout)

        top_controls = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addStretch(1)
        row1.addWidget(self.train_button)
        row1.addStretch(1)
        top_controls.addLayout(row1)
        top_controls.addWidget(data_source_box)

        progress_box = QGroupBox('Training Progress')
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.linear_bar)
        progress_layout.addWidget(self.gpr_bar)
        progress_layout.addWidget(self.dtc_bar)
        progress_box.setLayout(progress_layout)

        metrics_box = QGroupBox('Metrics (on saved test set)')
        metrics_layout = QVBoxLayout()
        # Table for metrics summary
        self.metrics_rows = ["R2", "RMSE", "MAE", "Accuracy", "F1", "Precision", "Recall"]
        self.metrics_table = QTableWidget(len(self.metrics_rows), 4)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Linear", "GPR", "DTC"])
        for r, name in enumerate(self.metrics_rows):
            self.metrics_table.setItem(r, 0, QTableWidgetItem(name))
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionMode(self.metrics_table.NoSelection)
        metrics_layout.addWidget(self.metrics_table)
        metrics_box.setLayout(metrics_layout)

        # Visualization row: 4 columns notion — metrics text (col 1) + combined plots (col 2-4)
        vis_box = QGroupBox('Visualization')
        vis_layout = QHBoxLayout()
        self.metrics_dashboard = MetricsDashboardCanvas(self)
        # add toolbar for zoom/pan via mouse
        toolbar_container = QVBoxLayout()
        toolbar = NavigationToolbar(self.metrics_dashboard, self)
        toolbar_container.addWidget(toolbar)
        toolbar_container.addWidget(self.metrics_dashboard)
        vis_layout.addWidget(metrics_box, 1)
        vis_layout.addLayout(toolbar_container, 3)
        vis_box.setLayout(vis_layout)

        logs_container = QVBoxLayout()
        logs_title = QLabel('Training Logs')
        logs_title.setStyleSheet('font-size:16px;font-weight:600;color:#1a2b4c;')
        logs_container.addWidget(logs_title)
        logs_container.addWidget(self.logs, stretch=1)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_controls)
        main_layout.addWidget(progress_box)
        main_layout.addWidget(vis_box)
        main_layout.addLayout(logs_container, stretch=2)

        # timers for simulation
        self._timer = QTimer(self)
        self._timer.setInterval(800)
        self._timer.timeout.connect(self._on_training_tick)
        self._epoch = 0

        # worker
        self._worker: TrainWorker = None
        # cache metrics for plotting
        self._metrics_linear = None
        self._metrics_gpr = None
        self._metrics_dtc = None
        # cache dtc model and train data for tree rendering
        self._dtc_model = None
        self._X_train_c = None
        self._y_train_c = None

    def preload_existing(self):
        """Load existing models and display their metrics. Check default version first."""
        # Reset metrics
        self._metrics_linear = None
        self._metrics_gpr = None
        self._metrics_dtc = None
        self.linear_metrics_lbl.setText('Linear => R2: --  RMSE: --  MAE: --')
        self.gpr_metrics_lbl.setText('GPR => R2: --  RMSE: --  MAE: --')
        self.dtc_metrics_lbl.setText('DTC => Acc: --  F1: --  Prec: --  Rec: --')
        self.linear_bar.setValue(0)
        self.gpr_bar.setValue(0)
        self.dtc_bar.setValue(0)
        tree_img = None
        
        # Check if any models exist (use default version)
        has_any_model = _exists_linear() or _exists_gpr() or _exists_dtc()
        if not has_any_model:
            self.logs.log("No trained models found. Please train models first.", 'info')
            return
        
        # Load Linear model metrics
        if _exists_linear():
            try:
                m = _compute_linear_metrics()
                if m:  # Only update if metrics exist
                    self._metrics_linear = m
                    # Safe conversion function
                    def safe_format_float(value, decimals=3):
                        try:
                            if value == '--':
                                return '--'
                            if isinstance(value, str):
                                value = float(value.replace(',', ''))
                            return f"{float(value):.{decimals}f}"
                        except (ValueError, TypeError):
                            return '--'

                    r2 = m.get('test_R2', m.get('R2', '--'))
                    rmse = m.get('test_RMSE', m.get('RMSE', '--'))
                    mae = m.get('test_MAE', m.get('MAE', '--'))

                self.linear_metrics_lbl.setText(
                    f"Linear => R2: {safe_format_float(r2)}  RMSE: {safe_format_float(rmse)}  MAE: {safe_format_float(mae)}"
                )
                self.linear_bar.setValue(100)
                self.logs.log("Loaded Linear model metrics from JSON", 'success')
            except Exception as e:
                self.logs.log(f"Failed to load Linear metrics: {e}", 'warning')
        
        # Load GPR model metrics
        if _exists_gpr():
            try:
                m = _compute_gpr_metrics()
                if m:  # Only update if metrics exist
                    self._metrics_gpr = m
                    # Safe conversion function
                    def safe_format_float(value, decimals=3):
                        try:
                            if value == '--':
                                return '--'
                            if isinstance(value, str):
                                value = float(value.replace(',', ''))
                            return f"{float(value):.{decimals}f}"
                        except (ValueError, TypeError):
                            return '--'

                    r2 = m.get('test_R2', m.get('R2', '--'))
                    rmse = m.get('test_RMSE', m.get('RMSE', '--'))
                    mae = m.get('test_MAE', m.get('MAE', '--'))

                self.gpr_metrics_lbl.setText(
                    f"GPR => R2: {safe_format_float(r2)}  RMSE: {safe_format_float(rmse)}  MAE: {safe_format_float(mae)}"
                )
                self.gpr_bar.setValue(100)
                self.logs.log("Loaded GPR model metrics from JSON", 'success')
            except Exception as e:
                self.logs.log(f"Failed to load GPR metrics: {e}", 'warning')
        
        # Load DTC model metrics
        if _exists_dtc():
            try:
                m = _compute_dtc_metrics()
                # Check for both old and new format metrics, or model existence
                has_old_metrics = any(k in m for k in ['Accuracy', 'F1', 'Precision', 'Recall']) if m else False
                has_new_metrics = any(k in m for k in ['test_Accuracy', 'test_F1', 'test_Precision', 'test_Recall']) if m else False
                has_model_path = 'model_path' in m if m else False

                if m and (has_old_metrics or has_new_metrics or has_model_path):
                    self._metrics_dtc = m
                    # Safe conversion function
                    def safe_format_float(value, decimals=3):
                        try:
                            if value == '--':
                                return '--'
                            if isinstance(value, str):
                                value = float(value.replace(',', ''))
                            return f"{float(value):.{decimals}f}"
                        except (ValueError, TypeError):
                            return '--'

                                        # Use test set metrics if available, otherwise fall back to legacy format
                    acc = m.get('test_Accuracy', m.get('Accuracy', '--'))
                    f1 = m.get('test_F1', m.get('F1', '--'))
                    prec = m.get('test_Precision', m.get('Precision', '--'))
                    rec = m.get('test_Recall', m.get('Recall', '--'))

                    # Check if we have real metrics (not None values)
                    has_real_values = all(v is not None and v != '--' for v in [acc, f1, prec, rec])

                    if has_real_values:
                        self.dtc_metrics_lbl.setText(
                            f"DTC => Acc: {safe_format_float(acc)}  F1: {safe_format_float(f1)}  Prec: {safe_format_float(prec)}  Rec: {safe_format_float(rec)}"
                        )
                        self.dtc_bar.setValue(100)
                        self.logs.log("Loaded DTC model metrics from JSON", 'success')
                    else:
                        # Has metrics but they are None/empty (skeleton metrics)
                        self.dtc_metrics_lbl.setText('DTC => Model exists (no metrics yet)')
                        self.dtc_bar.setValue(50)
                        self.logs.log("DTC model exists but metrics are empty", 'info')

                    # Always try to render tree image if we have model data
                    try:
                        self.logs.log("DEBUG: Attempting to render DTC tree image...", 'info')
                        tree_img = self._render_dtc_tree_png()
                        if tree_img is not None:
                            self.logs.log("DEBUG: DTC tree image rendered successfully", 'success')
                        else:
                            self.logs.log("DEBUG: DTC tree image render returned None", 'warning')
                    except Exception as e:
                        self.logs.log(f"Failed to render DTC tree: {e}", 'warning')
                        tree_img = None
                elif m:
                    # Has skeleton metrics but no real metrics
                    self._metrics_dtc = m
                    self.dtc_metrics_lbl.setText('DTC => Model available')
                    self.dtc_bar.setValue(50)
                    self.logs.log("DTC model exists (skeleton metrics only)", 'info')
                    # Try to render tree image
                    try:
                        self.logs.log("DEBUG: Attempting to render DTC tree image (skeleton metrics)...", 'info')
                        tree_img = self._render_dtc_tree_png()
                        if tree_img is not None:
                            self.logs.log("DEBUG: DTC tree image rendered successfully", 'success')
                        else:
                            self.logs.log("DEBUG: DTC tree image render returned None", 'warning')
                    except Exception as e:
                        self.logs.log(f"Failed to render DTC tree: {e}", 'warning')
                        tree_img = None
                else:
                    # No metrics at all
                    self.dtc_metrics_lbl.setText('DTC => No trained model')
                    self.dtc_bar.setValue(0)
                    self._metrics_dtc = {}
                    self.logs.log("DTC model exists but no metrics available", 'info')
            except Exception as e:
                self.logs.log(f"Failed to load DTC metrics: {e}", 'warning')
                self.dtc_metrics_lbl.setText('DTC => Load failed')
                self.dtc_bar.setValue(0)
                self._metrics_dtc = {}
        else:
            self.dtc_metrics_lbl.setText('DTC => No model available')
            self.dtc_bar.setValue(0)
            self._metrics_dtc = {}
            self.logs.log("No DTC model found - DTC will not be available", 'info')
        
        # Update dashboard and table
        self.metrics_dashboard.update_regression(self._metrics_linear, self._metrics_gpr)
        # Only update DTC dashboard if we have valid metrics
        # Debug: log the actual DTC metrics content
        if self._metrics_dtc:
            self.logs.log(f"DEBUG - DTC metrics content: {self._metrics_dtc}", 'info')

        # Check for both old and new format metrics, or even just model existence
        has_old_metrics = any(k in self._metrics_dtc for k in ['Accuracy', 'F1', 'Precision', 'Recall']) if self._metrics_dtc else False
        has_new_metrics = any(k in self._metrics_dtc for k in ['test_Accuracy', 'test_F1', 'test_Precision', 'test_Recall']) if self._metrics_dtc else False
        has_model_path = 'model_path' in self._metrics_dtc if self._metrics_dtc else False

        dtc_has_metrics = self._metrics_dtc and (has_old_metrics or has_new_metrics or has_model_path)

        self.logs.log(f"DEBUG - DTC metrics check: has_metrics={bool(self._metrics_dtc)}, old={has_old_metrics}, new={has_new_metrics}, model_path={has_model_path}, final={dtc_has_metrics}", 'info')

        if dtc_has_metrics:
            self.metrics_dashboard.update_classification(self._metrics_dtc, tree_img)
            self.logs.log(f"DTC dashboard updated with metrics: {list(self._metrics_dtc.keys())}", 'success')
        else:
            self.metrics_dashboard.update_classification({}, None)
            self.logs.log("DTC dashboard cleared - no valid metrics found", 'warning')
        self._update_metrics_table()

    def start_training(self):
        # Check artifacts existence
        has_artifacts = ARTIFACTS_DIR.exists() and any([
            _exists_linear(), _exists_gpr(), _exists_dtc()
        ])
        from_excel = self.data_source_combo.currentText() == 'From Excel'

        # UI 初始化
        self.train_button.setEnabled(False)
        self.train_button.setText('Training...')
        self.logs.clear()
        self.linear_bar.setValue(0)
        self.gpr_bar.setValue(0)
        self.dtc_bar.setValue(0)
        self.linear_metrics_lbl.setText('Linear => R2: --  RMSE: --  MAE: --')
        self.gpr_metrics_lbl.setText('GPR => R2: --  RMSE: --  MAE: --')
        self.dtc_metrics_lbl.setText('DTC => Acc: --  F1: --  Prec: --  Rec: --')

        if from_excel:
            # Excel 训练路径
            if not self.selected_excel_path:
                QMessageBox.warning(self, 'Data source', 'Please select an Excel file for training.')
                self.train_button.setEnabled(True)
                self.train_button.setText('Start Training')
                return

            # 如果已有权重，询问是否创建新版本
            if has_artifacts:
                reply = QMessageBox.question(self, 'Confirm',
                                             'Detected existing models. Create a new version by retraining from Excel?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply != QMessageBox.Yes:
                    self.train_button.setEnabled(True)
                    self.train_button.setText('Start Training')
                    return
                version_name = f"user_{int(time.time())}"
            else:
                # 第一次训练，使用默认版本名
                version_name = DEFAULT_VERSION

            self.logs.log(f"Creating model version: {version_name}")
            # Run local Excel training
            self._worker = ExcelTrainWorker(self.selected_excel_path, version_name, self)
        else:
            # From Weights：不训练，只加载已有权重并显示。如果没有，则提示切换到 From Excel。
            if not has_artifacts:
                QMessageBox.information(self, 'Weights not found',
                                        'No weights found in artifacts. Please switch to "From Excel" and select a file to train.')
                self.train_button.setEnabled(True)
                self.train_button.setText('Start Training')
                return
            # 加载显示现有权重指标
            self.logs.log('Using existing weights (no training). Loading metrics...', 'info')
            self.preload_existing()
            self.train_button.setEnabled(True)
            self.train_button.setText('Start Training')
            return

        # 连接并启动线程（仅 Excel 路径会走到这里）
        self._worker.progress.connect(self._on_progress)
        self._worker.metrics_ready.connect(self._on_metrics)
        self._worker.log.connect(self.logs.log)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.start()

    def _on_training_tick(self):
        self._epoch += 1
        train_acc = min(0.6 + 0.04 * self._epoch + random.uniform(-0.01, 0.01), 0.98)
        valid_acc = max(train_acc - random.uniform(0.01, 0.05), 0.55)
        # keep lightweight stats simulation
        self.logs.log(f"Heartbeat epoch {self._epoch}...", 'info')
        self.acc_value.setText(f"{valid_acc:.3f}")
        self.loss_value.setText(f"{(1 - valid_acc):.2f}")
        self.epoch_value.setText(str(self._epoch))
        self.step_value.setText(f"{random.randint(20, 60)}ms")

        if self._epoch >= 10:
            self._timer.stop()
            self.logs.log('UI heartbeat finished.', 'success')

    @QtCore.pyqtSlot(str, int)
    def _on_progress(self, model_key: str, percent: int):
        if model_key == 'linear':
            self.linear_bar.setValue(percent)
        elif model_key == 'gpr':
            self.gpr_bar.setValue(percent)
        else:
            self.dtc_bar.setValue(percent)

    @QtCore.pyqtSlot(str, dict)
    def _on_metrics(self, model_key: str, metrics: Dict[str, float]):
        if model_key == 'linear':
            # Use test set metrics for display (new format with test_/train_ prefixes)
            r2 = metrics.get('test_R2', metrics.get('R2', '--'))
            rmse = metrics.get('test_RMSE', metrics.get('RMSE', '--'))
            mae = metrics.get('test_MAE', metrics.get('MAE', '--'))

            # Safe conversion function
            def safe_format_float(value, decimals=3):
                try:
                    if value == '--':
                        return '--'
                    if isinstance(value, str):
                        value = float(value.replace(',', ''))
                    return f"{float(value):.{decimals}f}"
                except (ValueError, TypeError):
                    return '--'

            self.linear_metrics_lbl.setText(
                f"Linear => R2: {safe_format_float(r2)}  RMSE: {safe_format_float(rmse)}  MAE: {safe_format_float(mae)}"
                if metrics else 'Linear => R2: --  RMSE: --  MAE: --'
            )
            self._metrics_linear = metrics
            self.metrics_dashboard.update_regression(self._metrics_linear, self._metrics_gpr)
        elif model_key == 'gpr':
            # Use test set metrics for display (new format with test_/train_ prefixes)
            r2 = metrics.get('test_R2', metrics.get('R2', '--'))
            rmse = metrics.get('test_RMSE', metrics.get('RMSE', '--'))
            mae = metrics.get('test_MAE', metrics.get('MAE', '--'))

            # Safe conversion function
            def safe_format_float(value, decimals=3):
                try:
                    if value == '--':
                        return '--'
                    if isinstance(value, str):
                        value = float(value.replace(',', ''))
                    return f"{float(value):.{decimals}f}"
                except (ValueError, TypeError):
                    return '--'

            self.gpr_metrics_lbl.setText(
                f"GPR => R2: {safe_format_float(r2)}  RMSE: {safe_format_float(rmse)}  MAE: {safe_format_float(mae)}"
                if metrics else 'GPR => R2: --  RMSE: --  MAE: --'
            )
            self._metrics_gpr = metrics
            self.metrics_dashboard.update_regression(self._metrics_linear, self._metrics_gpr)
        else:
            if metrics:
                # Use test set metrics for display (new format with test_/train_ prefixes)
                acc = metrics.get('test_Accuracy', metrics.get('Accuracy', '--'))
                f1 = metrics.get('test_F1', metrics.get('F1', '--'))
                prec = metrics.get('test_Precision', metrics.get('Precision', '--'))
                rec = metrics.get('test_Recall', metrics.get('Recall', '--'))

                # Safe conversion function
                def safe_format_float(value, decimals=3):
                    try:
                        if value == '--':
                            return '--'
                        if isinstance(value, str):
                            value = float(value.replace(',', ''))
                        return f"{float(value):.{decimals}f}"
                    except (ValueError, TypeError):
                        return '--'

                self.dtc_metrics_lbl.setText(
                    f"DTC => Acc: {safe_format_float(acc)}  F1: {safe_format_float(f1)}  Prec: {safe_format_float(prec)}  Rec: {safe_format_float(rec)}"
                )
            else:
                self.dtc_metrics_lbl.setText('DTC => Acc: --  F1: --  Prec: --  Rec: --')
            self._metrics_dtc = metrics
            # log dtc metrics for troubleshooting
            try:
                self.logs.log('[DTC metrics] ' + json.dumps(metrics, ensure_ascii=False), 'info')
            except Exception:
                self.logs.log('[DTC metrics] ' + str(metrics), 'info')
            # try render decision tree to image
            tree_img = None
            try:
                tree_img = self._render_dtc_tree_png()
            except Exception as e:
                self.logs.log(f"Render decision tree failed: {e}")
            self.metrics_dashboard.update_classification(self._metrics_dtc, tree_img)
        # update table summary after any metrics update
        self._update_metrics_table()

    def _render_dtc_tree_png(self, version: str = DEFAULT_VERSION):
        """Render DTC as PNG. Prefer graphviz (with custom class colors), fallback to sklearn.plot_tree."""
        # Load model and metadata
        try:
            dtc_dir = get_dtc_dir(version)
            print(f"DEBUG: Loading DTC model from {dtc_dir}")
            dtc_model = joblib.load(dtc_dir / 'model.joblib')
            meta_path = dtc_dir / 'meta.json'
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                feat_names = meta.get('feature_names', KEY_FEATURES)
            else:
                feat_names = KEY_FEATURES
            class_names = [str(c) for c in getattr(dtc_model, 'classes_', [])]
            print(f"DEBUG: Model loaded successfully, features: {len(feat_names)}, classes: {len(class_names)}")
        except Exception as e:
            print(f"DEBUG: Failed to load model/metadata: {e}")
            return None

        # Try graphviz first
        try:
            print("DEBUG: Trying graphviz...")
            import graphviz
            dot = export_graphviz(
                dtc_model,
                out_file=None,
                feature_names=feat_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                special_characters=True,
                impurity=False,
            )
            # Increase font size
            dot = re.sub(r'fontsize=([0-9]+)', 'fontsize=24', dot)
            # Recolor nodes by majority class using provided color map; also display class index+1
            try:
                lines = dot.split('\n')
                new_lines = []
                for line in lines:
                    if re.match(r'^\s*\d+\s*\[', line):
                        value_match = re.search(r'value\s*=\s*\[([\d\s,]+)\]', line)
                        if value_match:
                            values = list(map(int, value_match.group(1).strip().split(',')))
                            if values and max(values) > 0:
                                class_idx = int(np.argmax(values))
                                class_name = str(class_idx)
                                color = DTC_CLASS_COLOR_DICT_STR.get(class_name, '#FFFFFF')
                                if 'fillcolor' not in line:
                                    line = line.replace(']', f', fillcolor="{color}", style=filled]')
                                else:
                                    line = re.sub(r'fillcolor="[^"]+"', f'fillcolor="{color}"', line)
                        # class=n -> class=n+1 for display
                        line = re.sub(r'class\s*=\s*(\d+)', lambda m: f'class = {int(m.group(1)) + 1}', line)
                    new_lines.append(line)
                dot = '\n'.join(new_lines)
            except Exception:
                pass

            src = graphviz.Source(dot)
            import io
            from PIL import Image
            png_bytes = src.pipe(format='png')
            image = Image.open(io.BytesIO(png_bytes)).convert('RGB')
            print("DEBUG: Graphviz rendering successful")
            return np.asarray(image)
        except Exception as e:
            print(f"DEBUG: Graphviz failed: {e}")
            # Fallback to sklearn's plot_tree with proper backend setup
            try:
                print("DEBUG: Trying sklearn fallback with proper backend...")

                # Force matplotlib backend before any matplotlib imports
                import sys
                if 'matplotlib' in sys.modules:
                    print("DEBUG: Matplotlib already imported, trying alternative approach...")
                    # If matplotlib is already imported, we need to use a different approach
                    import matplotlib
                    matplotlib.use('Agg', force=True)

                # Import after backend is set
                import matplotlib.pyplot as plt
                import io
                from PIL import Image
                from sklearn import tree as sktree

                # Create figure with explicit backend
                with plt.ioff():  # Turn off interactive mode
                    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

                    try:
                        sktree.plot_tree(
                            dtc_model,
                            feature_names=feat_names,
                            class_names=class_names,
                            filled=True,
                            rounded=True,
                            impurity=False,
                            fontsize=10,
                            ax=ax,
                        )

                        # Save with proper buffer handling
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight',
                                   facecolor='white', edgecolor='none')
                        buf.seek(0)
                        image = Image.open(buf).convert('RGB')
                        result = np.asarray(image)

                        print(f"DEBUG: Sklearn fallback successful: {result.shape}")
                        return result

                    finally:
                        plt.close(fig)  # Always close the figure

            except Exception as e2:
                print(f"DEBUG: Sklearn fallback failed: {e2}")
                # Try manual tree visualization
                try:
                    print("DEBUG: Trying manual tree visualization...")

                    import matplotlib.pyplot as plt
                    import io
                    from PIL import Image
                    import textwrap

                    # Get tree structure manually
                    tree = dtc_model.tree_
                    n_nodes = tree.node_count
                    children_left = tree.children_left
                    children_right = tree.children_right
                    feature = tree.feature
                    threshold = tree.threshold
                    value = tree.value

                    with plt.ioff():
                        fig, ax = plt.subplots(figsize=(12, 8))

                        def add_node_text(node_id, x, y, level=0, max_level=3):
                            if node_id == -1 or level > max_level:
                                return

                            # Calculate position
                            x_pos = x
                            y_pos = 1.0 - (level * 0.25)

                            # Node information
                            if feature[node_id] >= 0:  # Internal node
                                feat_idx = feature[node_id]
                                feat_name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"feature_{feat_idx}"
                                node_text = f"{feat_name}\n≤ {threshold[node_id]:.3f}"

                                # Add class prediction
                                class_idx = np.argmax(value[node_id][0])
                                samples = np.sum(value[node_id][0])
                                node_text += f"\nClass {class_idx + 1}\n({int(samples)} samples)"
                            else:  # Leaf node
                                class_idx = np.argmax(value[node_id][0])
                                samples = np.sum(value[node_id][0])
                                node_text = f"LEAF\nClass {class_idx + 1}\n({int(samples)} samples)"

                            # Draw node box
                            ax.add_patch(plt.Rectangle((x_pos-0.15, y_pos-0.08), 0.3, 0.16,
                                                     fill=True, facecolor='lightblue', alpha=0.7,
                                                     edgecolor='black', linewidth=1))
                            ax.text(x_pos, y_pos, node_text, ha='center', va='center',
                                   fontsize=9, wrap=True)

                            # Draw children recursively
                            if level < max_level and children_left[node_id] != -1:
                                # Left child
                                add_node_text(children_left[node_id], x_pos - 0.25, y_pos, level + 1, max_level)
                                # Right child
                                add_node_text(children_right[node_id], x_pos + 0.25, y_pos, level + 1, max_level)

                                # Draw connecting lines
                                left_y = 1.0 - ((level + 1) * 0.25)
                                right_y = left_y
                                ax.plot([x_pos, x_pos - 0.25], [y_pos - 0.08, left_y + 0.08],
                                       'k-', linewidth=1, alpha=0.7)
                                ax.plot([x_pos, x_pos + 0.25], [y_pos - 0.08, right_y + 0.08],
                                       'k-', linewidth=1, alpha=0.7)

                        # Start from root
                        add_node_text(0, 0.5, 1.0)

                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1.1)
                        ax.axis('off')
                        ax.set_title('Decision Tree Structure', fontsize=14, fontweight='bold', pad=20)

                        # Save
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight',
                                   facecolor='white', dpi=120)
                        plt.close(fig)
                        buf.seek(0)
                        image = Image.open(buf).convert('RGB')
                        result = np.asarray(image)

                        print(f"DEBUG: Manual tree visualization successful: {result.shape}")
                        return result

                except Exception as e3:
                    print(f"DEBUG: Manual visualization failed: {e3}")

                    # Last resort: simple information display
                    try:
                        print("DEBUG: Using simple information display...")

                        import matplotlib.pyplot as plt
                        import io
                        from PIL import Image

                        with plt.ioff():
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Get tree info
                            tree = dtc_model.tree_
                            n_nodes = tree.node_count
                            n_leaves = tree.n_leaves_ if hasattr(tree, 'n_leaves_') else 'Unknown'
                            max_depth = dtc_model.get_depth() if hasattr(dtc_model, 'get_depth') else 'Unknown'

                            info_text = f"""
Decision Tree Summary

Total Nodes: {n_nodes}
Leaf Nodes: {n_leaves}
Maximum Depth: {max_depth}
Features: {len(feat_names)}
Classes: {len(class_names)}

Features Used:
{chr(10).join(f"• {name}" for name in feat_names)}

Classes:
{chr(10).join(f"• Class {i+1}" for i in range(len(class_names)))}

Note: Install Graphviz for detailed tree visualization
"""

                            ax.text(0.5, 0.5, info_text, ha='center', va='center',
                                   fontsize=11, transform=ax.transAxes,
                                   fontfamily='monospace', linespacing=1.5)
                            ax.axis('off')
                            ax.set_title('Decision Tree Information', fontsize=14, fontweight='bold')

                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight',
                                       facecolor='white', dpi=100)
                            plt.close(fig)
                            buf.seek(0)
                            image = Image.open(buf).convert('RGB')
                            result = np.asarray(image)

                            print(f"DEBUG: Simple info display successful: {result.shape}")
                            return result

                    except Exception as e4:
                        print(f"DEBUG: All visualization methods failed: {e4}")
                        return None
    def _update_metrics_table(self):
        # Initialize all cells
        def set_cell(row_idx: int, col_idx: int, val):
            # Safe conversion for display
            if val is None:
                txt = "--"
            else:
                try:
                    if isinstance(val, str):
                        val = float(val.replace(',', ''))
                    if isinstance(val, (int, float)):
                        txt = f"{val:.3f}"
                    else:
                        txt = str(val)
                except (ValueError, TypeError):
                    txt = "--"

            item = self.metrics_table.item(row_idx, col_idx)
            if item is None:
                item = QTableWidgetItem(txt)
                self.metrics_table.setItem(row_idx, col_idx, item)
            else:
                item.setText(txt)

        # Clear model columns first
        for r in range(len(self.metrics_rows)):
            for c in (1, 2, 3):
                set_cell(r, c, None)

        # Fill Linear
        if self._metrics_linear:
            set_cell(0, 1, self._metrics_linear.get('test_R2', self._metrics_linear.get('R2')))
            set_cell(1, 1, self._metrics_linear.get('test_RMSE', self._metrics_linear.get('RMSE')))
            set_cell(2, 1, self._metrics_linear.get('test_MAE', self._metrics_linear.get('MAE')))

        # Fill GPR
        if self._metrics_gpr:
            set_cell(0, 2, self._metrics_gpr.get('test_R2', self._metrics_gpr.get('R2')))
            set_cell(1, 2, self._metrics_gpr.get('test_RMSE', self._metrics_gpr.get('RMSE')))
            set_cell(2, 2, self._metrics_gpr.get('test_MAE', self._metrics_gpr.get('MAE')))

        # Fill DTC
        if self._metrics_dtc:
            set_cell(3, 3, self._metrics_dtc.get('test_Accuracy', self._metrics_dtc.get('Accuracy')))
            set_cell(4, 3, self._metrics_dtc.get('test_F1', self._metrics_dtc.get('F1')))
            set_cell(5, 3, self._metrics_dtc.get('test_Precision', self._metrics_dtc.get('Precision')))
            set_cell(6, 3, self._metrics_dtc.get('test_Recall', self._metrics_dtc.get('Recall')))

    @QtCore.pyqtSlot()
    def _on_finished(self):
        self.train_button.setEnabled(True)
        self.train_button.setText('Start Training')
        # 如果是 TrainWorker，并且失败标记为 True，则提示失败
        if isinstance(self._worker, TrainWorker) and getattr(self._worker, 'failed', False):
            self.logs.log('Training finished with errors. See logs above.', 'error')
            QMessageBox.warning(self, 'Training Failed', 'Training encountered errors. Please check logs.')
        else:
            self.logs.log('All training finished.', 'success')

class PredictionSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Inputs
        self.model_select = QComboBox()
        self.model_select.addItems(['linear', 'gpr', 'dtc', 'all'])

        self.feature_inputs: List[QLineEdit] = []
        form_grid = QGridLayout()
        # Version selector for weights
        self.version_combo = QComboBox()
        self.refresh_versions()
        self.refresh_versions_btn = QPushButton('Refresh Versions')
        self.refresh_versions_btn.clicked.connect(self.refresh_versions)

        # Model selection - Row 0
        form_grid.addWidget(QLabel('Model'), 0, 0)
        form_grid.addWidget(self.model_select, 0, 1)

        # Model Version selection - Row 1
        form_grid.addWidget(QLabel('Model Version'), 1, 0)
        version_row = QHBoxLayout()
        version_row.addWidget(self.version_combo)
        version_row.addWidget(self.refresh_versions_btn)
        version_row.addStretch(1)
        form_grid.addLayout(version_row, 1, 1)

        # Feature inputs - Starting from Row 2
        row = 2
        col = 0
        self.example_values = _load_example_values()
        for idx, feat in enumerate(KEY_FEATURES):
            lbl = QLabel(feat)
            inp = QLineEdit()
            default_str = str(self.example_values.get(feat, 0.0))
            inp.setPlaceholderText(default_str)
            inp.setText(default_str)
            self.feature_inputs.append(inp)
            form_grid.addWidget(lbl, row, col)
            form_grid.addWidget(inp, row, col + 1)
            if col == 0:
                col = 2
            else:
                col = 0
                row += 1

        self.predict_button = QPushButton('Run Prediction')
        self.predict_button.setCursor(Qt.PointingHandCursor)
        self.predict_button.setStyleSheet('background:#3498db;color:white;font-weight:600;padding:10px 20px;border-radius:6px;')
        self.predict_button.clicked.connect(self.on_predict)

        # Outputs (left panel)
        self.pred_label = QLabel('')  # highlighted predicted value summary
        self.pred_label.setStyleSheet('font-size:18px;font-weight:800;color:#e67e22;')
        self.output_title = QLabel('')  # equation for linear, or short summary
        self.output_title.setStyleSheet('font-size:14px;font-weight:600;color:#1a2b4c;')
        self.std_label = QLabel('')  # for GPR std
        self.std_label.setStyleSheet('font-size:12px;color:#1a2b4c;')
        self.ci_label = QLabel('')  # for Linear/GPR CI via bootstrap on saved test
        self.ci_label.setStyleSheet('font-size:12px;color:#1a2b4c;')
        # Probability display for DTC (text format)
        self.proba_label = QLabel('')
        self.proba_label.setStyleSheet('font-family: monospace; font-size: 10px; color: #1a2b4c; background-color: #f8f9fa; padding: 10px; border-radius: 5px;')
        self.proba_label.setWordWrap(True)
        self.proba_label.setText("Class Probabilities:\n(No prediction yet)")

        # Preview area for DTC predicted class PDF (middle panel)
        self.class_preview_title = QLabel('')
        self.class_preview_title.setStyleSheet('font-size:13px;font-weight:600;color:#1a2b4c;')
        class ZoomableGraphicsView(QGraphicsView):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._zoom = 0
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
            def wheelEvent(self, event):
                if event.angleDelta().y() > 0:
                    factor = 1.15
                    self._zoom += 1
                else:
                    factor = 1/1.15
                    self._zoom -= 1
                if self._zoom < -10:
                    self._zoom = -10
                self.scale(factor, factor)
        # create scene and view
        self.class_scene = QGraphicsScene()
        self.class_view = ZoomableGraphicsView(self.class_scene)
        self.class_view.setRenderHints(self.class_view.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.class_view.setDragMode(QGraphicsView.ScrollHandDrag)

        # Right panel: DTC decision path
        self.dtc_path_box = QGroupBox('DTC Path')
        # Zoomable graphics view for path image
        self.dtc_path_scene = QGraphicsScene()
        self.dtc_path_view = ZoomableGraphicsView(self.dtc_path_scene)
        self.dtc_path_view.setRenderHints(self.dtc_path_view.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.dtc_path_view.setDragMode(QGraphicsView.ScrollHandDrag)
        # Fallback text when graph cannot be rendered
        self.dtc_path_text = QPlainTextEdit()
        self.dtc_path_text.setReadOnly(True)
        self.dtc_path_text.setStyleSheet("font-family: 'Courier New'; font-size: 12px;")
        path_layout = QVBoxLayout()
        path_layout.addWidget(self.dtc_path_view)
        path_layout.addWidget(self.dtc_path_text)
        self.dtc_path_box.setLayout(path_layout)

        self.logs = LogConsole(self)

        # Layout
        form_box = QGroupBox('Input Parameters')
        form_layout = QVBoxLayout()
        form_layout.addLayout(form_grid)
        form_box.setLayout(form_layout)

        # Logs panel
        logs_box = QGroupBox('Logs')
        logs_layout = QVBoxLayout()
        logs_layout.addWidget(self.logs)
        logs_box.setLayout(logs_layout)

        # Left: Prediction Output
        left_box = QGroupBox('Prediction Output')
        left_layout = QVBoxLayout()

        # Create scroll area for output content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMaximumHeight(400)  # Set reasonable max height

        # Create container widget for scrollable content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.addWidget(self.pred_label)
        scroll_layout.addWidget(self.output_title)
        scroll_layout.addWidget(self.std_label)
        scroll_layout.addWidget(self.ci_label)
        scroll_layout.addWidget(self.proba_label)
        scroll_layout.addStretch(1)  # Add stretch to push content up

        scroll_area.setWidget(scroll_content)
        left_layout.addWidget(scroll_area)
        left_box.setLayout(left_layout)

        # Middle: Predicted Class Preview
        middle_box = QGroupBox('Predicted Class Preview')
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.class_preview_title)
        middle_layout.addWidget(self.class_view)
        middle_box.setLayout(middle_layout)

        # Three columns horizontally: left (output), middle (preview), right (path)
        columns = QHBoxLayout()
        columns.addWidget(left_box, 2)
        columns.addWidget(middle_box, 2)
        columns.addWidget(self.dtc_path_box, 2)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(form_box)
        # Define top controls (predict button row)
        top_controls = QHBoxLayout()
        top_controls.addStretch(1)
        top_controls.addWidget(self.predict_button)
        top_controls.addStretch(1)
        main_layout.addLayout(top_controls)
        main_layout.addLayout(columns, stretch=3)
        main_layout.addWidget(logs_box, stretch=2)

        # events
        self.model_select.currentTextChanged.connect(self._on_model_change)
        self._on_model_change(self.model_select.currentText())

    def refresh_versions(self):
        """Refresh the list of available model versions."""
        self.version_combo.clear()
        versions = get_available_versions()
        if versions:
            self.version_combo.addItems(versions)
            self.version_combo.setEnabled(True)
            # 优先选择默认版本，如果不存在则选择第一个可用版本
            if DEFAULT_VERSION in versions:
                self.version_combo.setCurrentText(DEFAULT_VERSION)
            else:
                self.version_combo.setCurrentIndex(0)
        else:
            self.version_combo.addItem("No models found")
            self.version_combo.setEnabled(False)
        # logs may not be initialized at construction time
        if hasattr(self, 'logs') and self.logs is not None:
            self.logs.log(f"Found {len(versions)} model versions: {versions}", 'info')

    def _get_current_version(self) -> str:
        """Get currently selected version."""
        return self.version_combo.currentText() if self.version_combo.currentText() != "No models found" else DEFAULT_VERSION

    def _dir_for(self, model: str) -> Path:
        version = self._get_current_version()
        if model == 'linear':
            return get_linear_dir(version)
        if model == 'gpr':
            return get_gpr_dir(version)
        if model == 'dtc':
            return get_dtc_dir(version)
        return get_version_dir(version)

    def _on_model_change(self, model: str):
        # toggle widgets visibility depending on model
        self.pred_label.setVisible(True)
        self.output_title.setVisible(model == 'linear')
        self.std_label.setVisible(model == 'gpr')
        self.ci_label.setVisible(model in ('linear', 'gpr', 'all'))

        # Check if DTC model exists in current version or default version
        current_has_dtc = (self._dir_for('dtc') / 'model.joblib').exists()
        default_has_dtc = (get_dtc_dir(DEFAULT_VERSION) / 'model.joblib').exists()
        dtc_available = current_has_dtc or default_has_dtc

        if model in ('dtc', 'all'):
            if not dtc_available:
                self.logs.log("DTC model not available in current or default version", 'warning')
                self.proba_label.setVisible(True)
                self.proba_label.setText("DTC model not available")
                self.class_preview_title.setVisible(False)
                self.class_view.setVisible(False)
                self.dtc_path_box.setVisible(False)
            else:
                self.proba_label.setVisible(True)
                self.proba_label.setText("Class Probabilities:\n(No prediction yet)")
                self.class_preview_title.setVisible(True)
                self.class_view.setVisible(True)
                self.dtc_path_box.setVisible(True)
                if not current_has_dtc and default_has_dtc:
                    self.logs.log("DTC model will use default version", 'info')
        else:
            self.proba_label.setVisible(False)
            self.class_preview_title.setVisible(False)
            self.class_view.setVisible(False)
            self.dtc_path_box.setVisible(False)

    def _parse_inputs(self) -> List[float]:
        values = []
        for i, inp in enumerate(self.feature_inputs):
            txt = inp.text().strip()
            if txt == '':
                raise ValueError(f"Input '{KEY_FEATURES[i]}' is empty")
            try:
                v = float(txt)
            except Exception:
                raise ValueError(f"Input '{KEY_FEATURES[i]}' must be a number")
            values.append(v)
        return values

    def _load_meta(self, dir_path: Path):
        meta = json.loads((dir_path / 'meta.json').read_text(encoding='utf-8'))
        feature_names = meta.get('feature_names', KEY_FEATURES)
        return feature_names

    def _predict_linear(self, nums: List[float]):
        dir_path = self._dir_for('linear')
        if not ((dir_path / 'model.pkl').exists() and (dir_path / 'scaler.joblib').exists()):
            raise FileNotFoundError('Linear model not found. Please train first.')
        feature_names = self._load_meta(dir_path)
        import joblib
        scaler = joblib.load(dir_path / 'scaler.joblib')
        lin_model = sm.load(str(dir_path / 'model.pkl'))
        # build dataframe in the order of feature_names
        df = pd.DataFrame([nums], columns=KEY_FEATURES)
        df = df[[c for c in feature_names]]
        X_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
        y_pred = lin_model.predict(sm.add_constant(X_scaled, has_constant='add'))
        # equation string from params
        try:
            params = lin_model.params
            const = float(params.get('const', params.iloc[0]))
            terms = []
            for name in feature_names:
                coef = float(params.get(name, 0.0))
                terms.append(f"{coef:.3f} * {name}")
            eq = "y = " + f"{const:.4f} + " + " + ".join(terms)
        except Exception:
            eq = "Linear equation unavailable"
        return float(y_pred.iloc[0]), eq

    def _predict_gpr(self, nums: List[float]):
        dir_path = self._dir_for('gpr')
        if not ((dir_path / 'model.joblib').exists() and (dir_path / 'scaler.joblib').exists()):
            raise FileNotFoundError('GPR model not found. Please train first.')
        import joblib
        feature_names = self._load_meta(dir_path)
        scaler = joblib.load(dir_path / 'scaler.joblib')
        gpr = joblib.load(dir_path / 'model.joblib')
        df = pd.DataFrame([nums], columns=KEY_FEATURES)
        df = df[[c for c in feature_names]]
        X_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
        y_pred, y_std = gpr.predict(X_scaled, return_std=True)
        return float(y_pred[0]), float(y_std[0])

    def _predict_dtc(self, nums: List[float]):
        dir_path = self._dir_for('dtc')
        dtc_model_path = dir_path / 'model.joblib'
        using_default = False

        # 如果当前版本没有DTC模型，尝试使用默认版本
        if not dtc_model_path.exists():
            default_dir_path = get_dtc_dir(DEFAULT_VERSION)
            if (default_dir_path / 'model.joblib').exists():
                dir_path = default_dir_path
                dtc_model_path = dir_path / 'model.joblib'
                using_default = True
                self.logs.log("Current version has no DTC model, using default version", 'warning')
            else:
                raise FileNotFoundError('DTC model not found in current or default version.')

        if using_default:
            self.logs.log("Using DTC model from default weights for prediction", 'info')

        import joblib
        model = joblib.load(dtc_model_path)
        meta = json.loads((dir_path / 'meta.json').read_text(encoding='utf-8'))
        feature_names = meta.get('feature_names', KEY_FEATURES)
        # order features robustly: prefer model.feature_names_in_
        order = getattr(model, 'feature_names_in_', None)
        if order is None:
            order = feature_names
        df = pd.DataFrame([nums], columns=KEY_FEATURES)
        cols = [c for c in order if c in df.columns]
        if len(cols) != len(order):
            # try use provided feature_names
            cols = [c for c in feature_names if c in df.columns]
        X_used = df[cols]
        # bypass feature-name checks if necessary
        try:
            proba = model.predict_proba(X_used)
        except Exception:
            proba = model.predict_proba(X_used.to_numpy())
        classes = getattr(model, 'classes_', np.arange(proba.shape[1]))
        return classes.tolist(), proba[0].tolist()

    def _show_dtc_pdf_preview(self, best_class):
        # Build path to Class/cluster_{cls}.pdf
        try:
            cls_name = str(int(best_class)) if isinstance(best_class, (int, np.integer, float)) else str(best_class)
        except Exception:
            cls_name = str(best_class)
        pdf_path = Path('Class') / f"cluster_{cls_name}.pdf"
        self.class_preview_title.setText(f"Predicted Class Preview: cluster_{cls_name}.pdf")
        if not pdf_path.exists():
            self.class_scene.clear()
            self.class_scene.addText(f"File not found: {pdf_path}")
            return
        # Try to render first page via PyMuPDF
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            qimg = QImage.fromData(img_bytes)
            self.class_scene.clear()
            self.class_scene.addPixmap(QPixmap.fromImage(qimg))
            doc.close()
        except Exception as e:
            # Fallback: open externally
            self.class_scene.clear()
            self.class_scene.addText("Opening PDF externally...")
            try:
                os.startfile(str(pdf_path))
            except Exception:
                self.class_scene.addText(f"Cannot preview. Open manually: {pdf_path}")

        # 显示使用的是哪个版本的模型
        current_version = self._get_current_version()
        if not (self._dir_for('dtc') / 'model.joblib').exists():
            self.logs.log(f"PDF preview using DTC model from default version (current version: {current_version} has no DTC model)", 'info')

    def _show_dtc_path(self, best_class, nums: List[float]):
        # Compute path taken by the DTC for this input, render with graphviz
        try:
            import joblib
            dtc_dir = self._dir_for('dtc')
            dtc_model_path = dtc_dir / 'model.joblib'
            using_default = False

            # 如果当前版本没有DTC模型，使用默认版本
            if not dtc_model_path.exists():
                default_dtc_dir = get_dtc_dir(DEFAULT_VERSION)
                if (default_dtc_dir / 'model.joblib').exists():
                    dtc_dir = default_dtc_dir
                    dtc_model_path = dtc_dir / 'model.joblib'
                    using_default = True
                    self.logs.log("Using default DTC model for path visualization", 'info')
                else:
                    raise FileNotFoundError("DTC model not found in current or default version")

            model = joblib.load(dtc_model_path)
        except Exception as e:
            self.dtc_path_text.setPlainText(f"Failed to load DTC model: {e}")
            self.dtc_path_view.setVisible(False)
            self.dtc_path_text.setVisible(True)
            return
        # Build one-row DataFrame in model order
        order = getattr(model, 'feature_names_in_', None)
        if order is None:
            order = KEY_FEATURES
        df = pd.DataFrame([nums], columns=KEY_FEATURES)
        cols = [c for c in order if c in df.columns]
        X_row = df[cols].iloc[0]
        tree = model.tree_
        feature = tree.feature
        threshold = tree.threshold
        children_left = tree.children_left
        children_right = tree.children_right
        node = 0
        nodes = []
        edges = []
        # Traverse and record nodes in the path
        while node != -1 and children_left[node] != -1:
            feat_idx = feature[node]
            feat_name = cols[feat_idx] if 0 <= feat_idx < len(cols) else f"f{feat_idx}"
            thr = threshold[node]
            xval = float(X_row.get(feat_name, np.nan))
            go_left = xval <= thr
            direction = '≤' if go_left else '>'
            label = f"{feat_name} {direction} {thr:.3f}\nx={xval:.3f}"
            nodes.append((node, label))
            next_node = children_left[node] if go_left else children_right[node]
            edges.append((node, next_node, 'True' if go_left else 'False'))
            node = next_node
        # Leaf
        leaf_node = node
        leaf_label = "leaf"
        pred_idx = None
        if leaf_node != -1:
            val = tree.value[leaf_node][0]
            pred_idx = int(np.argmax(val))
            leaf_label = f"leaf\nvalue={val.tolist()}\nclass={pred_idx+1}"
        nodes.append((leaf_node, leaf_label))

        # Try render graph via graphviz
        try:
            import graphviz
            dot = ["digraph Path {", 'rankdir=TB;', 'node [shape=box, style="rounded,filled", fontsize=10];']
            # define nodes
            for nid, lab in nodes:
                fill = '#e8f0fe' if nid != leaf_node else '#d1ffd6'
                dot.append(f'"{nid}" [label="{lab}", fillcolor="{fill}"];')
            # define edges
            for src, dst, elab in edges:
                dot.append(f'"{src}" -> "{dst}" [label="{elab}"];')
            dot.append('}')
            dot_str = "\n".join(dot)
            src = graphviz.Source(dot_str)
            import io
            from PIL import Image
            png_bytes = src.pipe(format='png')
            img = QImage.fromData(png_bytes)
            self.dtc_path_scene.clear()
            self.dtc_path_scene.addPixmap(QPixmap.fromImage(img))
            self.dtc_path_view.fitInView(self.dtc_path_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.dtc_path_view.setVisible(True)
            self.dtc_path_text.setVisible(False)
        except Exception as e:
            # Fallback to text path
            lines = []
            for nid, lab in nodes:
                lines.append(f"node {nid}: {lab}")
            if pred_idx is not None:
                lines.append(f"leaf {leaf_node}: class={pred_idx+1}")
            self.dtc_path_text.setPlainText("\n".join(lines))
            self.dtc_path_view.setVisible(False)
            self.dtc_path_text.setVisible(True)

    def _bootstrap_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, n_iterations: int = 500, ci: float = 0.95):
        rng = np.random.default_rng(seed=42)
        n = len(y_true)
        r2s, rmses, maes = [], [], []
        for _ in range(n_iterations):
            idx = rng.choice(n, size=n, replace=True)
            yt = y_true[idx]
            yp = y_pred[idx]
            r2s.append(r2_score(yt, yp))
            rmses.append(float(np.sqrt(mean_squared_error(yt, yp))))
            maes.append(float(mean_absolute_error(yt, yp)))
        def ci_bounds(vals):
            lower = float(np.percentile(vals, (1 - ci) / 2 * 100))
            upper = float(np.percentile(vals, (1 + ci) / 2 * 100))
            return float(np.mean(vals)), lower, upper
        return {
            'R2': ci_bounds(r2s),
            'RMSE': ci_bounds(rmses),
            'MAE': ci_bounds(maes),
        }

    def _compute_ci_text(self, model_key: str) -> str:
        """Load and display complete model performance metrics including CI"""
        try:
            version = self._get_current_version()

            # Load metrics from JSON
            if model_key == 'linear':
                metrics = _compute_linear_metrics(version)
            elif model_key == 'gpr':
                metrics = _compute_gpr_metrics(version)
            else:
                return ""

            if not metrics:
                return "Metrics: Not available"

            lines = []

            if model_key in ['linear', 'gpr']:
                # Regression metrics
                lines.append("==>> Model Performance:")

                # Helper function to safely convert to float
                def safe_float(value, default=0.0):
                    try:
                        if isinstance(value, str):
                            return float(value.replace(',', ''))
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                # Test set metrics
                if 'test_R2' in metrics and 'test_RMSE' in metrics and 'test_MAE' in metrics:
                    lines.append("Test Set:")
                    lines.append(f"  R²: {safe_float(metrics['test_R2']):.4f}")
                    lines.append(f"  RMSE: {safe_float(metrics['test_RMSE']):.4f}")
                    lines.append(f"  MAE: {safe_float(metrics['test_MAE']):.4f}")

                # Train set metrics
                if 'train_R2' in metrics and 'train_RMSE' in metrics and 'train_MAE' in metrics:
                    lines.append("Train Set:")
                    lines.append(f"  R²: {safe_float(metrics['train_R2']):.4f}")
                    lines.append(f"  RMSE: {safe_float(metrics['train_RMSE']):.4f}")
                    lines.append(f"  MAE: {safe_float(metrics['train_MAE']):.4f}")

                # CI information
                if 'test_ci' in metrics and 'train_ci' in metrics:
                    lines.append("")
                    lines.append("~  Confidence Intervals (95%):")

                    test_ci = metrics['test_ci']
                    train_ci = metrics['train_ci']

                    if 'R2' in test_ci and 'R2' in train_ci:
                        test_r2_vals = test_ci['R2']
                        train_r2_vals = train_ci['R2']
                        if len(test_r2_vals) >= 3 and len(train_r2_vals) >= 3:
                            test_r2_mean, test_r2_low, test_r2_high = [safe_float(v) for v in test_r2_vals[:3]]
                            train_r2_mean, train_r2_low, train_r2_high = [safe_float(v) for v in train_r2_vals[:3]]
                            lines.append(f"  R²: Test={test_r2_mean:.4f}±{(test_r2_high-test_r2_low)/2:.4f}")
                            lines.append(f"      Train={train_r2_mean:.4f}±{(train_r2_high-train_r2_low)/2:.4f}")

                    if 'RMSE' in test_ci and 'RMSE' in train_ci:
                        test_rmse_vals = test_ci['RMSE']
                        train_rmse_vals = train_ci['RMSE']
                        if len(test_rmse_vals) >= 3 and len(train_rmse_vals) >= 3:
                            test_rmse_mean, test_rmse_low, test_rmse_high = [safe_float(v) for v in test_rmse_vals[:3]]
                            train_rmse_mean, train_rmse_low, train_rmse_high = [safe_float(v) for v in train_rmse_vals[:3]]
                            lines.append(f"  RMSE: Test={test_rmse_mean:.4f}±{(test_rmse_high-test_rmse_low)/2:.4f}")
                            lines.append(f"        Train={train_rmse_mean:.4f}±{(train_rmse_high-train_rmse_low)/2:.4f}")

                    if 'MAE' in test_ci and 'MAE' in train_ci:
                        test_mae_vals = test_ci['MAE']
                        train_mae_vals = train_ci['MAE']
                        if len(test_mae_vals) >= 3 and len(train_mae_vals) >= 3:
                            test_mae_mean, test_mae_low, test_mae_high = [safe_float(v) for v in test_mae_vals[:3]]
                            train_mae_mean, train_mae_low, train_mae_high = [safe_float(v) for v in train_mae_vals[:3]]
                            lines.append(f"  MAE: Test={test_mae_mean:.4f}±{(test_mae_high-test_mae_low)/2:.4f}")
                            lines.append(f"       Train={train_mae_mean:.4f}±{(train_mae_high-train_mae_low)/2:.4f}")

                # Equation for Linear
                if model_key == 'linear' and 'equation' in metrics:
                    lines.append("")
                    lines.append(f"~  Equation: {metrics['equation']}")

            elif model_key == 'dtc':
                # Classification metrics
                lines.append("==>> Model Performance:")

                # Test set metrics
                if 'test_Accuracy' in metrics and 'test_F1' in metrics:
                    lines.append("Test Set:")
                    lines.append(f"  Accuracy: {safe_float(metrics.get('test_Accuracy', 0)):.4f}")
                    lines.append(f"  F1: {safe_float(metrics.get('test_F1', 0)):.4f}")
                    lines.append(f"  Precision: {safe_float(metrics.get('test_Precision', 0)):.4f}")
                    lines.append(f"  Recall: {safe_float(metrics.get('test_Recall', 0)):.4f}")
                    if 'test_AUC' in metrics:
                        lines.append(f"  AUC: {safe_float(metrics.get('test_AUC', 0)):.4f}")

                # Train set metrics
                if 'train_Accuracy' in metrics and 'train_F1' in metrics:
                    lines.append("Train Set:")
                    lines.append(f"  Accuracy: {safe_float(metrics.get('train_Accuracy', 0)):.4f}")
                    lines.append(f"  F1: {safe_float(metrics.get('train_F1', 0)):.4f}")
                    lines.append(f"  Precision: {safe_float(metrics.get('train_Precision', 0)):.4f}")
                    lines.append(f"  Recall: {safe_float(metrics.get('train_Recall', 0)):.4f}")

            return '\n'.join(lines)

        except Exception as e:
            return f"Error loading metrics: {str(e)}"

    def on_predict(self):
        model = self.model_select.currentText()
        self.logs.clear()
        try:
            nums = self._parse_inputs()
        except Exception as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))
            self.logs.log(f"Invalid input: {e}", 'error')
            return

        # Reset outputs
        self.pred_label.setText('')
        self.output_title.setText('')
        self.std_label.setText('')
        self.ci_label.setText('')
        self.class_preview_title.setText('')
        self.class_scene.clear()
        self.dtc_path_text.clear()
        self.dtc_path_scene.clear()
        self.proba_label.setText("Class Probabilities:\n(No prediction yet)")

        try:
            if model == 'linear':
                y, eq = self._predict_linear(nums)
                self.pred_label.setText(f"y_pred: {y:.6f}")
                self.output_title.setText(eq)
                self.logs.log(f"[Linear] y_pred = {y:.6f}", 'success')
                try:
                    ci_text = self._compute_ci_text('linear')
                    self.ci_label.setText(ci_text)
                except Exception as e:
                    self.logs.log(f"Linear CI failed: {e}", 'warning')
            elif model == 'gpr':
                y, std = self._predict_gpr(nums)
                self.pred_label.setText(f"y_pred: {y:.6f}")
                self.std_label.setText(f"Std: {std:.6f}")
                self.logs.log(f"[GPR] y_pred = {y:.6f}, std = {std:.6f}", 'success')
                try:
                    ci_text = self._compute_ci_text('gpr')
                    self.ci_label.setText(ci_text)
                except Exception as e:
                    self.logs.log(f"GPR CI failed: {e}", 'warning')
            elif model == 'dtc':
                classes, proba = self._predict_dtc(nums)
                # Format probabilities as text
                proba_text = "Class Probabilities:\n"
                proba_text += "=" * 30 + "\n"
                for i in range(len(classes)):
                    cls_name = classes[i]
                    cls_name = int(cls_name) + 1 # 类别从1开始
                    prob = proba[i]
                    # Create a visual bar chart
                    bar_length = int(prob * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    proba_text += f"Class {cls_name:2d}: {prob:.4f} | {bar}\n"
                proba_text += "=" * 30
                self.proba_label.setText(proba_text)

                if proba:
                    best_idx = int(np.argmax(proba))
                    disp_cls = classes[best_idx]
                    try:
                        disp_cls_int = int(disp_cls)
                        disp_cls_show = disp_cls_int + 1
                    except Exception:
                        disp_cls_show = disp_cls
                    self.pred_label.setText(f"Predicted class: {disp_cls_show} (p={proba[best_idx]:.4f})")
                    self._show_dtc_pdf_preview(disp_cls_show)
                    self._show_dtc_path(disp_cls, nums)
                self.logs.log(f"[DTC] classes = {classes}; proba = {proba}", 'success')
            else:
                # run all three
                y_lin, eq = self._predict_linear(nums)
                y_gpr, std = self._predict_gpr(nums)
                classes, proba = self._predict_dtc(nums)
                # combined highlighted line
                if proba:
                    best_idx = int(np.argmax(proba))
                    dtc_text = f"DTC: cls {classes[best_idx]+ 1} (p={proba[best_idx]:.4f})"
                else:
                    dtc_text = "DTC: n/a"
                self.pred_label.setText(f"Linear: {y_lin:.6f} | GPR: {y_gpr:.6f} | {dtc_text}")
                self.output_title.setText(eq)
                self.std_label.setText(f"Std: {std:.6f}")
                # Format probabilities as text for "all" mode
                proba_text = "Class Probabilities:\n"
                proba_text += "=" * 30 + "\n"
                for i in range(len(classes)):
                    cls_name = classes[i]
                    cls_name = int(cls_name) + 1 # 类别从1开始
                    prob = proba[i]
                    # Create a visual bar chart
                    bar_length = int(prob * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    proba_text += f"Class {cls_name:2d}: {prob:.4f} | {bar}\n"
                proba_text += "=" * 30
                self.proba_label.setText(proba_text)
                self.logs.log(f"[All] Linear={y_lin:.6f}, GPR={y_gpr:.6f} (std={std:.6f}), DTC={proba}", 'success')
                # show pdf preview for best class
                if proba:
                    disp_cls = classes[best_idx]
                    try:
                        disp_cls_int = int(disp_cls)
                        disp_cls_show = disp_cls_int + 1
                    except Exception:
                        disp_cls_show = disp_cls
                    self._show_dtc_pdf_preview(disp_cls_show)
                    self._show_dtc_path(disp_cls, nums)
                # CI for both
                try:
                    ci_lin = self._compute_ci_text('linear')
                except Exception as e:
                    ci_lin = f"Linear CI failed: {e}"
                try:
                    ci_gpr = self._compute_ci_text('gpr')
                except Exception as e:
                    ci_gpr = f"GPR CI failed: {e}"
                self.ci_label.setText(f"{ci_lin} | {ci_gpr}")
        except FileNotFoundError as e:
            QMessageBox.warning(self, 'Model Missing', str(e))
            self.logs.log(str(e), 'error')
        except Exception as e:
            QMessageBox.critical(self, 'Prediction Failed', str(e))
            self.logs.log(f"Prediction failed: {e}", 'error')


class Header(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background:#1a2b4c;color:white;')

        #
        self.app_title = QLabel('AILD')
        self.app_title.setStyleSheet('font-weight:700;font-size:16px;')

        self.train_btn = QPushButton('Training')
        self.predict_btn = QPushButton('Prediction')
        for btn in (self.train_btn, self.predict_btn):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setCheckable(True)
            btn.setStyleSheet('background:#2c3e66;color:white;padding:8px 16px;border-radius:4px;')

        layout = QHBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.predict_btn)


class Footer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background:#1a2b4c;color:#e2e8f0;')
        self.status_light = QLabel()
        self.status_light.setFixedSize(12, 12)
        self.status_light.setStyleSheet('background:#2ecc71;border-radius:6px;')
        self.status_label = QLabel('Status: Training - Ready')
        self.version_label = QLabel('Version: v1.0.0 | © 2025 AILD')

        layout = QHBoxLayout(self)
        left = QHBoxLayout()
        left.addWidget(self.status_light)
        left.addWidget(self.status_label)
        left.addStretch(1)
        right = QHBoxLayout()
        right.addWidget(self.version_label)
        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=2)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AILD - Desktop')
        self.resize(1280, 840)

        self.header = Header(self)
        self.footer = Footer(self)

        self.training_section = TrainingSection(self)
        self.prediction_section = PredictionSection(self)

        self.stacked = QStackedWidget(self)
        self.stacked.addWidget(self.training_section)
        self.stacked.addWidget(self.prediction_section)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self.header)
        layout.addWidget(self.stacked, stretch=1)
        layout.addWidget(self.footer)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        # connections
        self.header.train_btn.clicked.connect(lambda: self.switch_mode(0, 'Training - Ready'))
        self.header.predict_btn.clicked.connect(lambda: self.switch_mode(1, 'Prediction - Ready'))

        # default
        self.header.train_btn.setChecked(True)

    def switch_mode(self, index: int, status_text: str):
        self.stacked.setCurrentIndex(index)
        for btn in (self.header.train_btn, self.header.predict_btn):
            btn.setChecked(False)
            btn.setStyleSheet('background:#2c3e66;color:white;padding:8px 16px;border-radius:4px;')
        if index == 0:
            self.header.train_btn.setChecked(True)
            self.header.train_btn.setStyleSheet('background:#3498db;color:white;padding:8px 16px;border-radius:4px;')
            # Preload existing models if available
            try:
                self.training_section.preload_existing()
            except Exception as e:
                self.footer.status_label.setText(f'Status: Training - Ready (preload failed: {e})')
        elif index == 1:
            self.header.predict_btn.setChecked(True)
            self.header.predict_btn.setStyleSheet('background:#3498db;color:white;padding:8px 16px;border-radius:4px;')

        self.footer.status_label.setText(f'Status: {status_text}')


def test_dtc_rendering():
    """Test function to verify DTC tree rendering works"""
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import make_classification
        import numpy as np

        print("Testing DTC rendering methods...")

        # Create a simple test model
        X, y = make_classification(n_samples=100, n_features=4, n_classes=3, random_state=42)
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        # Test rendering methods
        feat_names = [f'feature_{i}' for i in range(4)]
        class_names = [f'class_{i}' for i in range(3)]

        # Test our improved rendering method
        try:
            # Simulate the rendering process without graphviz
            print("DEBUG: Testing sklearn fallback with proper backend...")

            # Force matplotlib backend before any matplotlib imports
            import sys
            if 'matplotlib' in sys.modules:
                print("DEBUG: Matplotlib already imported, trying alternative approach...")
                # If matplotlib is already imported, we need to use a different approach
                import matplotlib
                matplotlib.use('Agg', force=True)

            # Import after backend is set
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            from sklearn import tree as sktree

            # Create figure with explicit backend
            with plt.ioff():  # Turn off interactive mode
                fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

                try:
                    sktree.plot_tree(
                        model,
                        feature_names=feat_names,
                        class_names=class_names,
                        filled=True,
                        rounded=True,
                        impurity=False,
                        fontsize=10,
                        ax=ax,
                    )

                    # Save with proper buffer handling
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    buf.seek(0)
                    image = Image.open(buf).convert('RGB')
                    result = np.asarray(image)

                    print(f"✓ Sklearn fallback test successful: {result.shape}")
                    return True

                finally:
                    plt.close(fig)  # Always close the figure

        except Exception as e:
            print(f"✗ Sklearn fallback test failed: {e}")

            # Test manual visualization
            try:
                print("DEBUG: Testing manual tree visualization...")

                import matplotlib.pyplot as plt
                import io
                from PIL import Image

                # Get tree structure manually
                tree = model.tree_
                n_nodes = tree.node_count
                children_left = tree.children_left
                children_right = tree.children_right
                feature = tree.feature
                threshold = tree.threshold
                value = tree.value

                with plt.ioff():
                    fig, ax = plt.subplots(figsize=(12, 8))

                    def add_node_text(node_id, x, y, level=0, max_level=3):
                        if node_id == -1 or level > max_level:
                            return

                        # Calculate position
                        x_pos = x
                        y_pos = 1.0 - (level * 0.25)

                        # Node information
                        if feature[node_id] >= 0:  # Internal node
                            feat_idx = feature[node_id]
                            feat_name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"feature_{feat_idx}"
                            node_text = f"{feat_name}\n≤ {threshold[node_id]:.3f}"

                            # Add class prediction
                            class_idx = np.argmax(value[node_id][0])
                            samples = np.sum(value[node_id][0])
                            node_text += f"\nClass {class_idx + 1}\n({int(samples)} samples)"
                        else:  # Leaf node
                            class_idx = np.argmax(value[node_id][0])
                            samples = np.sum(value[node_id][0])
                            node_text = f"LEAF\nClass {class_idx + 1}\n({int(samples)} samples)"

                        # Draw node box
                        ax.add_patch(plt.Rectangle((x_pos-0.15, y_pos-0.08), 0.3, 0.16,
                                                 fill=True, facecolor='lightblue', alpha=0.7,
                                                 edgecolor='black', linewidth=1))
                        ax.text(x_pos, y_pos, node_text, ha='center', va='center',
                               fontsize=9, wrap=True)

                        # Draw children recursively
                        if level < max_level and children_left[node_id] != -1:
                            # Left child
                            add_node_text(children_left[node_id], x_pos - 0.25, y_pos, level + 1, max_level)
                            # Right child
                            add_node_text(children_right[node_id], x_pos + 0.25, y_pos, level + 1, max_level)

                            # Draw connecting lines
                            left_y = 1.0 - ((level + 1) * 0.25)
                            right_y = left_y
                            ax.plot([x_pos, x_pos - 0.25], [y_pos - 0.08, left_y + 0.08],
                                   'k-', linewidth=1, alpha=0.7)
                            ax.plot([x_pos, x_pos + 0.25], [y_pos - 0.08, right_y + 0.08],
                                   'k-', linewidth=1, alpha=0.7)

                    # Start from root
                    add_node_text(0, 0.5, 1.0)

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1.1)
                    ax.axis('off')
                    ax.set_title('Decision Tree Structure', fontsize=14, fontweight='bold', pad=20)

                    # Save
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight',
                               facecolor='white', dpi=120)
                    plt.close(fig)
                    buf.seek(0)
                    image = Image.open(buf).convert('RGB')
                    result = np.asarray(image)

                    print(f"✓ Manual tree visualization test successful: {result.shape}")
                    return True

            except Exception as e2:
                print(f"✗ Manual visualization test also failed: {e2}")
                return False

    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        return False


def main():
    # Check if test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test-dtc':
        success = test_dtc_rendering()
        sys.exit(0 if success else 1)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


