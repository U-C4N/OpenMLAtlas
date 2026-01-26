<p align="center">
  <img src="assets/demo.gif" alt="OpenMLAtlas Demo" width="800"/>
</p>

<h1 align="center">OpenMLAtlas</h1>

<p align="center">
  <b>A Comprehensive Machine Learning Algorithm Atlas with Hands-on Python Implementations</b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-algorithms">Algorithms</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## Overview

**OpenMLAtlas** is an educational repository designed to help beginners and intermediate learners understand machine learning algorithms through clean, well-commented Python implementations. Each algorithm comes with:

- Detailed code comments explaining every step
- Real-world analogies to make concepts intuitive
- Sample datasets for hands-on practice
- Visualizations to understand model behavior

Whether you're a student, self-learner, or developer transitioning into ML, this atlas provides a solid foundation for understanding core machine learning concepts.

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
├── video/                      # Remotion video project
│   ├── src/
│   └── out/
│
├── convert_to_gif.py           # Video to GIF converter
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/U-C4N/OpenMLAtlas.git
   cd OpenMLAtlas
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

3. **Run any algorithm**
   ```bash
   cd CORE-ML/Linear-regression
   python lineer.py
   ```

## Usage

Each algorithm directory is self-contained. Navigate to any directory and run the Python file:

```bash
# Linear Regression
cd CORE-ML/Linear-regression
python lineer.py

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

### Example Output

**Linear Regression:**
```
   YearsExperience   Salary
0              1.1  39343.0
1              1.3  46205.0
2              1.5  37731.0
3              2.0  43525.0
4              2.2  39891.0
R2 Score: 0.9749...
```

**Random Forest:**
```
Prediction: [1]  # 1 = Hire, 0 = Don't hire
```

## Key Concepts Covered

### Supervised Learning
- **Regression**: Predicting continuous values (salary, price, temperature)
- **Classification**: Predicting categories (spam/not spam, hired/not hired)

### Model Evaluation
- **R² Score**: How well the model explains variance
- **Confusion Matrix**: Visualizing classification performance
- **ROC/AUC**: Evaluating binary classifiers

### Data Preprocessing
- **Feature Scaling**: StandardScaler for normalization
- **Label Encoding**: Converting categorical to numerical
- **Train/Test Split**: Preventing overfitting

### Reinforcement Learning
- **Exploration vs Exploitation**: The fundamental trade-off
- **UCB Algorithm**: Balancing known rewards with uncertainty

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `matplotlib` | Visualization |
| `scikit-learn` | ML algorithms |
| `statsmodels` | Statistical models (optional) |

Install all at once:
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
```

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-algorithm`)
3. **Add** your implementation with detailed comments
4. **Test** your code with sample data
5. **Submit** a pull request

### Guidelines
- Follow the existing code style
- Add comprehensive comments explaining each step
- Include a sample dataset if applicable
- Update README with new algorithm info

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
