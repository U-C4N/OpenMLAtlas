<p align="center">
  <img src="assets/demo.gif" alt="OpenMLAtlas Demo" width="800"/>
</p>

<h1 align="center">OpenMLAtlas</h1>

<p align="center">
  <b>A Comprehensive Machine Learning Algorithm Atlas with Hands-on Python Implementations</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#algorithms">Algorithms</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#usage">Usage</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Overview

**OpenMLAtlas** is an educational repository designed to help beginners and intermediate learners understand machine learning algorithms through clean, beginner-friendly Python implementations.

What you get in this repo:

- Clear, step-by-step Python scripts (one folder per topic)
- Small sample datasets (`.csv`) to practice with
- Simple visualizations to see model behavior

## Quick Start

1. Clone the repository

```bash
git clone https://github.com/U-C4N/OpenMLAtlas.git
cd OpenMLAtlas
```

2. Install dependencies (recommended)

```bash
pip install -r requirements.txt
```

3. Run your first example

```bash
cd CORE-ML/Linear-regression
python lineer.py
```

## Algorithms

### Regression

| Algorithm | Description | Directory |
|-----------|-------------|-----------|
| **Linear Regression** | Simple regression with one feature | [`CORE-ML/Linear-regression`](CORE-ML/Linear-regression) |
| **Multiple Linear Regression** | Regression with multiple features | [`CORE-ML/Multiple-linear-regression`](CORE-ML/Multiple-linear-regression) |
| **Polynomial Regression** | Captures non-linear relationships | [`CORE-ML/Polynomial-linear-regression`](CORE-ML/Polynomial-linear-regression) |
| **Support Vector Regression (SVR)** | Robust regression using support vectors | [`CORE-ML/Support-vector-regression(SVR)`](CORE-ML/Support-vector-regression(SVR)) |

### Classification

| Algorithm | Description | Directory |
|-----------|-------------|-----------|
| **Logistic Regression** | Binary/multi-class classification | [`CORE-ML/Logistic-Regression`](CORE-ML/Logistic-Regression) |
| **K-Nearest Neighbors (KNN)** | Instance-based lazy learning | [`CORE-ML/KNN`](CORE-ML/KNN) |
| **Decision Tree** | Tree-based decision making | [`CORE-ML/Decision Tree`](CORE-ML/Decision%20Tree) |
| **Random Forest** | Ensemble of decision trees | [`CORE-ML/Random-Forest`](CORE-ML/Random-Forest) |

### Model Evaluation

| Topic | Description | Directory |
|-------|-------------|-----------|
| **Confusion Matrix** | Understanding TP, TN, FP, FN | [`CORE-ML/Model-Evaluation`](CORE-ML/Model-Evaluation) |
| **False Positive/Negative** | Type I and Type II errors | [`CORE-ML/Model-Evaluation`](CORE-ML/Model-Evaluation) |
| **Accuracy Paradox** | When accuracy misleads | [`CORE-ML/Model-Evaluation`](CORE-ML/Model-Evaluation) |
| **ROC Curve** | Receiver Operating Characteristic | [`CORE-ML/Model-Evaluation`](CORE-ML/Model-Evaluation) |
| **AUC Value** | Area Under the Curve | [`CORE-ML/Model-Evaluation`](CORE-ML/Model-Evaluation) |

### Reinforcement Learning

| Algorithm | Description | Directory |
|-----------|-------------|-----------|
| **Random Selection** | Baseline random approach | [`CORE-ML/Reinforcement-Learning`](CORE-ML/Reinforcement-Learning) |
| **Upper Confidence Bound (UCB)** | Exploration vs Exploitation | [`CORE-ML/Reinforcement-Learning`](CORE-ML/Reinforcement-Learning) |

## Project Structure

```
OpenMLAtlas/
├── CORE-ML/
│   ├── Linear-regression/
│   │   ├── lineer.py           # Simple linear regression
│   │   ├── multiple-lineer.py  # Extended example
│   │   └── salary.csv          # Sample dataset
│   │
│   ├── Multiple-linear-regression/
│   │   ├── mlr.py              # Multiple linear regression
│   │   ├── sm_ols.py           # Statsmodels OLS approach
│   │   ├── data.csv            # Sample dataset
│   │   └── analysis.md         # Detailed analysis
│   │
│   ├── Polynomial-linear-regression/
│   │   ├── new.py              # Polynomial regression
│   │   └── data.csv            # Sample dataset
│   │
│   ├── Support-vector-regression(SVR)/
│   │   ├── svr.py              # SVR implementation
│   │   └── data.csv            # Sample dataset
│   │
│   ├── Logistic-Regression/
│   │   ├── l-r.py              # Logistic regression
│   │   └── data.csv            # Sample dataset
│   │
│   ├── KNN/
│   │   └── knn.py              # K-Nearest Neighbors
│   │
│   ├── Decision Tree/
│   │   ├── dt.py               # Decision tree regressor
│   │   └── salary_data.csv     # Sample dataset
│   │
│   ├── Random-Forest/
│   │   ├── rf.py               # Random forest classifier
│   │   └── hiring_data.csv     # Sample dataset
│   │
│   ├── Model-Evaluation/
│   │   ├── confusion_matrix.py
│   │   ├── false_positive_negative.py
│   │   ├── accuracy_paradox.py
│   │   ├── roc_curve.py
│   │   └── auc_value.py
│   │
│   └── Reinforcement-Learning/
│       ├── 01_random_selection.py
│       ├── 02_ucb.py
│       └── ads_data.csv
│
├── assets/
│   └── demo.gif                # Project demo animation
│
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository

```bash
git clone https://github.com/U-C4N/OpenMLAtlas.git
cd OpenMLAtlas
```

2. Install dependencies

Recommended:

```bash
pip install -r requirements.txt
```

Alternative (manual):

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
```

## Usage

Each algorithm directory is self-contained. Navigate to any directory and run the Python file:

```bash
# Linear Regression
cd CORE-ML/Linear-regression
python lineer.py

# Multiple Linear Regression
cd CORE-ML/Multiple-linear-regression
python mlr.py

# Decision Tree
cd "CORE-ML/Decision Tree"
python dt.py

# Random Forest
cd CORE-ML/Random-Forest
python rf.py

# UCB Reinforcement Learning
cd CORE-ML/Reinforcement-Learning
python 02_ucb.py
```

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Add your implementation with clear, beginner-friendly comments
4. Test your code with sample data
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Scikit-learn documentation and tutorials
- Machine Learning courses that inspired this project
- The open-source community

---

<p align="center">
  <b>Happy Learning!</b><br>
  If this helped you, consider giving it a star
</p>
