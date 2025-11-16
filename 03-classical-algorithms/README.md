# 🎲 03 - Classical Machine Learning Algorithms

Welcome to the **Classical Algorithms** module of OpenMLAtlas! This is where you'll master the powerful, time-tested algorithms that form the core of traditional machine learning. From decision trees to support vector machines, these methods remain highly effective for many real-world problems.

## 🎓 Overview

This module covers advanced classical machine learning algorithms that go beyond basic linear models. You'll learn tree-based methods, support vector machines, clustering techniques, dimensionality reduction, and ensemble methods—all essential tools in any ML practitioner's toolkit.

## 🧩 Module Structure

| Directory | Topic | Description |
|-----------|-------|-------------|
| 🌳 `tree-based-models/` | **Tree-Based Models** | **Decision trees and their powerful extensions** |
| ↳ `decision-trees/` | Decision Trees | CART, ID3, C4.5 for classification and regression |
| ↳ `random-forest/` | Random Forests | Ensemble of decision trees using bagging |
| ↳ `gradient-boosting/` | Gradient Boosting | Sequential ensemble learning for high performance |
| ↳ `xgboost-lightgbm-catboost/` | Modern Boosting | State-of-the-art gradient boosting implementations |
| 🎯 `svm/` | Support Vector Machines | Maximum margin classifiers, kernels, SVR |
| 🔵 `clustering/` | **Clustering Algorithms** | **Unsupervised grouping and pattern discovery** |
| ↳ `kmeans/` | K-Means Clustering | Centroid-based clustering algorithm |
| ↳ `hierarchical-clustering/` | Hierarchical Clustering | Agglomerative and divisive clustering approaches |
| ↳ `dbscan/` | DBSCAN | Density-based clustering for arbitrary shapes |
| ↳ `gmm/` | Gaussian Mixture Models | Probabilistic clustering with EM algorithm |
| 📉 `dimensionality-reduction/` | **Dimensionality Reduction** | **Reducing features while preserving information** |
| ↳ `pca/` | Principal Component Analysis | Linear dimensionality reduction via eigenvectors |
| ↳ `lda/` | Linear Discriminant Analysis | Supervised dimensionality reduction |
| ↳ `manifold-learning-tsne-umap/` | Manifold Learning | t-SNE, UMAP for nonlinear dimensionality reduction |
| 🎪 `ensemble-methods/` | **Ensemble Methods** | **Combining multiple models for better performance** |
| ↳ `bagging/` | Bagging | Bootstrap aggregating to reduce variance |
| ↳ `boosting/` | Boosting | Sequential learning to reduce bias |
| ↳ `stacking/` | Stacking | Meta-learning by combining diverse models |

## 🗺️ Learning Path

We recommend following this order:

1. **tree-based-models/** - Start with tree-based methods
   - **decision-trees/** - Understand the foundation
   - **random-forest/** - Learn ensemble with bagging
   - **gradient-boosting/** - Master sequential ensemble learning
   - **xgboost-lightgbm-catboost/** - Apply modern implementations
2. **svm/** - Learn maximum margin classifiers and kernel methods
3. **clustering/** - Master unsupervised learning
   - **kmeans/** - Start with the most popular clustering algorithm
   - **hierarchical-clustering/** - Learn dendrogram-based approaches
   - **dbscan/** - Understand density-based clustering
   - **gmm/** - Explore probabilistic clustering
4. **dimensionality-reduction/** - Learn to handle high-dimensional data
   - **pca/** - Master the most common technique
   - **lda/** - Understand supervised reduction
   - **manifold-learning-tsne-umap/** - Explore nonlinear methods
5. **ensemble-methods/** - Combine everything you've learned
   - **bagging/** - Reduce variance through averaging
   - **boosting/** - Reduce bias through sequential learning
   - **stacking/** - Build meta-models for optimal performance

However, feel free to jump to specific topics based on your needs!

## 🔑 What You'll Learn

### 🌳 Tree-Based Methods
- **Decision Trees**: How to build interpretable models using recursive partitioning
- **Random Forests**: How bagging creates robust, high-performing ensembles
- **Gradient Boosting**: How sequential learning corrects errors iteratively
- **Modern Boosting**: XGBoost, LightGBM, CatBoost for production-grade performance

### 🎯 Support Vector Machines
- **Linear SVM**: Maximum margin classification for linearly separable data
- **Kernel Trick**: Mapping data to higher dimensions without explicit computation
- **Non-linear SVM**: RBF, polynomial, and custom kernels
- **Support Vector Regression**: Extending SVM to regression problems

### 🔵 Clustering Algorithms
- **K-Means**: Centroid-based partitioning and choosing optimal K
- **Hierarchical Clustering**: Building dendrograms and linkage methods
- **DBSCAN**: Density-based clustering for arbitrary-shaped clusters
- **GMM**: Probabilistic clustering with soft assignments

### 📉 Dimensionality Reduction
- **PCA**: Variance-preserving linear projection
- **LDA**: Class-separating linear projection
- **t-SNE & UMAP**: Nonlinear manifold learning for visualization and preprocessing

### 🎪 Ensemble Strategies
- **Bagging**: Reducing variance through bootstrap aggregation
- **Boosting**: Reducing bias through adaptive reweighting
- **Stacking**: Combining diverse models with meta-learners

## 📋 Prerequisites

Before starting this module, you should be familiar with:
- **Core ML Concepts**: Supervised vs unsupervised learning, model evaluation
- **Linear Models**: Linear and logistic regression, regularization
- **Probability & Statistics**: Distributions, statistical testing
- **Linear Algebra**: Matrix operations, eigenvalues/eigenvectors
- **Python/Scikit-learn**: Basic ML workflows
- Completed **[02-core-ml/](../02-core-ml/)** or equivalent knowledge

## 🎬 Getting Started

1. Ensure you have completed the prerequisites
2. Install required libraries: `pip install numpy pandas matplotlib scikit-learn xgboost lightgbm catboost jupyter`
3. Start with the recommended learning path or jump to a specific topic
4. Work through theory, code examples, and exercises in each subdirectory
5. Compare different algorithms on the same datasets to understand their strengths

## 📚 How to Use This Module

Each subdirectory contains:
- **Theory**: Concept explanations and mathematical foundations (`.md` files)
- **Jupyter Notebooks**: Interactive code examples with visualizations (`.ipynb` files)
- **Comparisons**: Side-by-side algorithm comparisons
- **Exercises**: Practice problems to reinforce your learning
- **Projects**: Hands-on projects to apply what you've learned
- **Resources**: Additional reading materials and references

### 🔧 Working with Jupyter Notebooks

To run the interactive examples:
```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab for a better experience
jupyter lab
```

Each notebook includes:
- 📝 Step-by-step algorithm implementations with explanations
- 📊 Interactive visualizations showing how algorithms work
- 🧪 Real datasets to experiment with different methods
- 💪 Hands-on exercises comparing algorithm performance
- 🎯 Practical tips for algorithm selection and tuning

## 🚀 Next Steps

Once you've mastered classical algorithms, move on to:
- **[04-deep-learning/](../04-deep-learning/)** - Neural networks and deep learning

## 💬 Contributing

Found an error? Have a suggestion? Feel free to open an issue or submit a pull request!

## 📜 License

This project is part of OpenMLAtlas - An open-source machine learning learning resource.

---

✨ **Remember**: These classical algorithms are still widely used in production! They're often faster to train, more interpretable, and require less data than deep learning methods. Master them well!
