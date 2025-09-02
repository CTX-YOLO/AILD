# AILD: Artificial Intelligence for Logical Descriptors

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-orange.svg)](https://pypi.org/project/PyQt5/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-red.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Read this in [Chinese/中文](README_CN.md)**

An intelligent desktop application for logical descriptors analysis in materials science, featuring machine learning model training and prediction capabilities with an intuitive graphical user interface.

## 🚀 Features

### Core Functionality
- **Multi-Model Support**: Linear Regression, Gaussian Process Regression (GPR), and Decision Tree Classification (DTC)
- **Interactive GUI**: User-friendly PyQt5-based interface for easy operation
- **Model Training**: Train models from Excel data or use pre-trained weights
- **Real-time Prediction**: Predict material properties with live model inference
- **Performance Visualization**: Comprehensive metrics dashboard and charts
- **Model Management**: Version control and artifact management system

### Advanced Capabilities
- **Bootstrap Confidence Intervals**: Statistical confidence analysis for predictions
- **SHAP Values**: Feature importance explanation and interpretability
- **Bayesian Optimization**: Hyperparameter tuning for Decision Tree models
- **Cross-Validation**: Robust model evaluation and validation
- **PDF Preview**: Visual preview of predicted material classes
- **Decision Tree Visualization**: Interactive tree structure visualization

## 📋 Requirements

- **Python**: 3.11 or higher
- **Operating System**: Windows 7/10/11, Linux, or macOS
- **RAM**: Minimum 2GB (4GB recommended)
- **Storage**: 1GB free space for models and data

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/CTX-YOLO/AILD.git
cd AILD
```

### 2. Install Dependencies
```bash
conda create -n AILD_env python=3.11.8 -y
conda activate AILD_env
pip install -r requirements.txt
```

### 3. Install Graphviz (for Decision Tree Visualization)
- **Windows**: Download from [Graphviz Official](https://graphviz.org/download/)[Note: set the environment variables] or `winget install Graphviz.Graphviz`
- **Ubuntu/Debian**: `sudo apt-get install graphviz`
- **macOS**: `brew install graphviz`

## 📖 Usage

### GUI Mode (Recommended)

1. **Launch the Application**
   ```bash
   python AILD.py
   ```

2. **Training Tab**:
   - Choose data source: "From Excel" or "From Weights"
   - Select Excel file (if using Excel data)
   - Click "Start Training" to train models
   - Monitor progress and view metrics in real-time

3. **Prediction Tab**:
   - Select model type (linear, gpr, dtc, or all)
   - Choose model version
   - Enter feature values or use example values
   - Click "Run Prediction" for instant results

### Command Line Mode

#### Training Models
```bash
# Train all models
python Train_and_Predictor.py --task all --mode train --excel-path your_data.xlsx

# Train specific model
python Train_and_Predictor.py --task linear --mode train --excel-path your_data.xlsx
python Train_and_Predictor.py --task gpr --mode train --excel-path your_data.xlsx
python Train_and_Predictor.py --task dtc --mode train --excel-path your_data.xlsx
```

#### Making Predictions
```bash
# Predict with specific model
python Train_and_Predictor.py --task linear --mode predict --values "2.7597,-1.3566,7.0,3.1857,1.5862,0.2068"

# Predict with all models
python Train_and_Predictor.py --task all --mode predict --values "2.7597,-1.3566,7.0,3.1857,1.5862,0.2068"
```

## 📊 Data Format

### Input Features (6 Features)
- `length`: Average first‑neighbor bond length
- `a_M1_dz2`:  dz²‑orbital center of the active metal
- `num`: First‑neighbor coordination number
- `a_M2_s`: Breadth of the s‑band on the first neighbor
- `a_M3_dxz`: Skewness of the dxz orbital on the first‑neighbor atom
- `c_Rs_dz2`: Fraction of ligand dz² in the projected density of states

### Training Data Format (Excel)
| length | a_M1_dz2 | num | a_M2_s | a_M3_dxz | c_Rs_dz2 | E_ad | cluster (optional) |
|--------|----------|-----|--------|----------|----------|------|-------------------|
| 2.7597 | -1.3566  | 7.0 | 3.1857 | 1.5862   | 0.2068   | 1.23 | 1                 |

### Prediction Input Format
Comma or space-separated values in order:
```
length,a_M1_dz2,num,a_M2_s,a_M3_dxz,c_Rs_dz2
```
Example: `2.7597,-1.3566,7.0,3.1857,1.5862,0.2068`

## 🏗️ Project Structure

```
AILD/
├── AILD.py                    # Main GUI application
├── Train_and_Predictor.py      # Command-line training and prediction
├── requirements.txt           # Python dependencies
├── README.md                  # English documentation
├── README_CN.md               # Chinese documentation
└── artifacts/                 # Model artifacts and weights
    ├── default_weights_files/ # Pre-trained models
    └── user_*/                # User-trained model versions
```

## 🔬 Models Overview

### 1. Linear Regression
- **Purpose**: Predict continuous target values (e.g., adsorption energy)
- **Features**: Equation generation, confidence intervals
- **Output**: Predicted value with uncertainty bounds

### 2. Gaussian Process Regression (GPR)
- **Purpose**: Probabilistic predictions with uncertainty quantification
- **Features**: Standard deviation output, kernel-based learning
- **Output**: Predicted value with confidence intervals

### 3. Decision Tree Classifier (DTC)
- **Purpose**: Material classification into discrete categories
- **Features**: Bayesian optimization, decision path visualization
- **Output**: Class probabilities and predicted category

## 📈 Performance Metrics

### Regression Models (Linear & GPR)
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Confidence Intervals**: 95% bootstrap confidence bounds

### Classification Model (DTC)
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Positive prediction accuracy
- **Recall**: Sensitivity/True positive rate
- **AUC**: Area under ROC curve

## 🎯 Key Features Explained

### Model Version Management
- Automatic version creation for new training sessions
- Fallback to default weights when user models unavailable
- Persistent artifact storage with metadata

### Real-time Visualization
- Interactive metrics dashboard with matplotlib
- Decision tree visualization with Graphviz
- Class-specific PDF preview functionality
- Zoomable and pannable graphics views

### Robust Error Handling
- Graceful degradation when models unavailable
- Comprehensive logging and user feedback
- Input validation and type checking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyQt5 for the desktop interface
- Powered by scikit-learn for machine learning algorithms
- Visualization support from matplotlib and Graphviz
- SHAP integration for model interpretability

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include your Python version, OS, and error messages

---

**AILD** - Advancing materials science through intelligent machine learning
