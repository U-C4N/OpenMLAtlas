# OpenMLAtlas - Project Summary

## Project Overview

**OpenMLAtlas** is a comprehensive visual machine learning tutorial repository designed for learners who prefer slow, clearly broken-down explanations. The project provides structured learning paths covering everything from mathematical foundations to practical ML deployment.

## Project Structure

The repository is organized into 6 major modules, progressing from foundational concepts to advanced practical applications:

### Module 1: Foundations (01-foundations/)
Mathematical and programming prerequisites for machine learning

### Module 2: Core Machine Learning (02-core-ml/)
Fundamental ML algorithms and concepts

### Module 3: Classical Algorithms (03-classical-algorithms/)
Tree-based models, SVMs, clustering, and ensemble methods

### Module 4: Deep Learning (04-deep-learning/)
Neural networks, CNNs, RNNs, transformers, and deep learning frameworks

### Module 5: Special Topics (05-special-topics/)
NLP, time series, recommender systems, generative models, reinforcement learning, and graph learning

### Module 6: Practical ML (06-practical-ml/)
Data preprocessing, pipelines, deployment, and MLOps

---

## Detailed Module Breakdown

## 01. Foundations Module

**Location**: `01-foundations/`
**Purpose**: Essential mathematical and programming prerequisites for machine learning

### Subfolders & Content

#### 1. math-basics/
- **Purpose**: Basic mathematical concepts and notation used throughout ML
- **Content**: Interactive web application for math practice
- **Features**:
  - Multilingual support (English, Spanish, French, Russian, Turkish)
  - Black-and-white UI design
  - Primary school level mathematics
  - Operations: Addition, Subtraction, Multiplication, Division
- **Files** (recently deleted):
  - `index.html` - Main HTML structure
  - `script.js` - Interactive functionality (229 lines)
  - `style.css` - Styling (182 lines)
- **Status**: Currently empty (files deleted from working directory)

#### 2. linear-algebra/
- **Purpose**: Vectors, matrices, eigenvalues, and matrix operations
- **Topics**: Essential for data representation in ML
- **Status**: Empty (awaiting content)

#### 3. calculus-for-ml/
- **Purpose**: Derivatives, gradients, and optimization concepts
- **Topics**: Understanding how models learn
- **Status**: Empty (awaiting content)

#### 4. probability-statistics/
- **Purpose**: Probability distributions, statistical inference, and hypothesis testing
- **Topics**: Core statistical concepts for ML
- **Status**: Empty (awaiting content)

#### 5. optimization/
- **Purpose**: Gradient descent, convex optimization, and loss function minimization
- **Topics**: Optimization techniques for model training
- **Status**: Empty (awaiting content)

#### 6. python-numpy-pandas/
- **Purpose**: Python programming fundamentals
- **Topics**:
  - NumPy for numerical computing
  - Pandas for data manipulation
- **Status**: Empty (awaiting content)

#### 7. visualization-basics/
- **Purpose**: Data visualization techniques
- **Topics**:
  - Matplotlib for plotting
  - Seaborn for exploratory analysis
- **Status**: Empty (awaiting content)

### Recommended Learning Path

The modules should be studied in sequence:
1. Math Basics
2. Linear Algebra
3. Calculus for ML
4. Probability & Statistics
5. Optimization
6. Python/NumPy/Pandas
7. Visualization Basics

### Learning Resources

The module includes curated resources:

**Books**:
- "Mathematics for Machine Learning" by Deisenroth, Faisal, and Ong
- "Linear Algebra and Its Applications" by Gilbert Strang
- "Calculus" by James Stewart
- "Introduction to Probability" by Blitzstein and Hwang
- "Python Data Science Handbook" by Jake VanderPlas
- "Think Stats" by Allen Downey

**Video Resources**:
- 3Blue1Brown (Linear Algebra & Calculus)
- StatQuest with Josh Starmer
- MIT OpenCourseWare 18.06
- Khan Academy
- Corey Schafer Python Tutorials

**Interactive Resources**:
- Immersive Linear Algebra (immersivemath.com)
- Seeing Theory (seeing-theory.brown.edu)
- Distill.pub
- GeoGebra
- Desmos
- Kaggle Learn

---

## 02. Core Machine Learning

**Location**: `02-core-ml/`

### Topics Covered:
- Supervised vs Unsupervised Learning
- Linear Regression
- Logistic Regression
- Regularization
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Model Evaluation
  - Regression Metrics
  - Classification Metrics
  - Cross Validation
- Feature Engineering
- Bias-Variance Tradeoff
- Overfitting & Underfitting

---

## 03. Classical Algorithms

**Location**: `03-classical-algorithms/`

### Tree-Based Models:
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost, LightGBM & CatBoost

### Support Vector Machines (SVM)

### Clustering:
- K-Means
- Gaussian Mixture Models (GMM)
- DBSCAN
- Hierarchical Clustering

### Dimensionality Reduction:
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Manifold Learning (t-SNE, UMAP)

### Ensemble Methods:
- Bagging
- Boosting
- Stacking

---

## 04. Deep Learning

**Location**: `04-deep-learning/`

### DL Foundations:
- Perceptron & MLP
- Backpropagation
- Initialization & Activations
- Optimization (SGD, Adam)
- Regularization (Dropout, BatchNorm)

### Computer Vision:
- CNN Basics
- Modern CNNs (ResNet, DenseNet)
- Transfer Learning

### Sequence Modeling:
- RNN, LSTM & GRU
- Sequence to Sequence
- Attention for Sequences

### Transformers & LLMs:
- Attention Basics
- Transformer Architecture
- LLM Overview

### Frameworks:
**PyTorch**:
- PyTorch Basics
- Tensors & Autograd
- Training Loops
- Datasets & DataLoaders
- Model Export & Deployment

**TensorFlow/Keras**:
- TF/Keras Basics
- TF Data Pipelines
- Custom Training Loops
- TF Serving & TF Lite

---

## 05. Special Topics

**Location**: `05-special-topics/`

### Natural Language Processing (NLP):
- Classic NLP (BoW, TF-IDF)
- Word Embeddings (Word2Vec, GloVe)
- Transformers in NLP

### Time Series:
- Time Series Basics
- ARIMA & Classical Models
- Feature-Based Approach
- Deep Learning for Time Series

### Recommender Systems:
- Implicit vs Explicit Feedback
- Collaborative Filtering
- Matrix Factorization
- Deep Learning Recommenders

### Generative Models:
- Autoencoders
- Variational Autoencoders (VAE)
- Generative Adversarial Networks (GANs)
- Diffusion Models

### Reinforcement Learning:
- RL Foundations & MDP
- Dynamic Programming
- Tabular Methods (Q-Learning, SARSA)
- Deep RL (DQN)
- Policy Gradient & Actor-Critic

### Graph Learning:
- Graph Theory Basics
- Graph Neural Networks

---

## 06. Practical ML

**Location**: `06-practical-ml/`

### Data Preprocessing:
- Missing Data
- Scaling & Normalization
- Categorical Encoding
- Outliers & Robust Methods
- Class Imbalance & Resampling

### Pipelines & Experimentation:
- Scikit-learn Pipelines
- Hyperparameter Optimization
- Experiment Tracking (MLflow)

### Deployment & Serving:
- Batch vs Online Serving
- REST API (FastAPI, Flask)
- Model Serialization & Versioning

### Monitoring & MLOps:
- Data Drift
- Model Drift
- Logging & Alerting
- CI/CD for ML

---

## Current Project Status

**Active Module**: 01-foundations
**Repository**: GitHub - OpenMLAtlas
**Main Branch**: main

### Recent Changes:
- Math basics interactive web application (HTML/CSS/JS) was created but recently deleted
- All foundation subfolders are currently empty and awaiting content
- Comprehensive README documentation is in place for all 6 modules
- Project structure and learning paths have been defined

### Git Status:
- Deleted files: `math-basics/index.html`, `math-basics/script.js`, `math-basics/style.css`
- Current branch: main
- Recent commits include README updates and module structure setup

---

## Project Philosophy

OpenMLAtlas follows a "slow learning" approach, prioritizing:
- **Visual explanations** over text-heavy tutorials
- **Clear breakdown** of complex concepts
- **Step-by-step progression** from basics to advanced topics
- **Practical examples** alongside theory
- **Comprehensive resources** for each topic

## Target Audience

This repository is designed for:
- Self-learners who prefer visual and interactive content
- Beginners starting their ML journey
- Practitioners who need to strengthen their fundamentals
- Anyone who learns best with clearly broken-down explanations

---

**Repository Path**: `C:\Users\ACER\Documents\GitHub\OpenMLAtlas\`
**Documentation Created**: November 21, 2025
