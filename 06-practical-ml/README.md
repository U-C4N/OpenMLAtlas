# 06 - Practical ML

## Overview

This module bridges the gap between ML theory and production systems. You'll learn essential skills for deploying ML models in real-world environments: data preprocessing at scale, building robust ML pipelines, model deployment and serving, monitoring systems, and MLOps best practices. These skills are critical for ML engineers and data scientists working in production environments.

## Learning Path (Folder Order)

Master production ML in this sequence:

1. **data-preprocessing/** - Production-grade data preparation
   - missing-data/ - Handling missing values (imputation, deletion strategies)
   - scaling-and-normalization/ - Feature scaling, standardization, normalization
   - categorical-encoding/ - One-hot, label encoding, target encoding, embeddings
   - outliers-and-robust-methods/ - Outlier detection and robust statistical methods
   - class-imbalance-resampling/ - SMOTE, undersampling, class weights

2. **pipelines-and-experimentation/** - Systematic ML development
   - sklearn-pipelines/ - Building reproducible preprocessing and training pipelines
   - hyperparameter-optimization/ - Grid search, random search, Bayesian optimization
   - experiment-tracking-mlflow/ - Tracking experiments, metrics, and artifacts with MLflow

3. **deployment-and-serving/** - Putting models into production
   - model-serialization-and-versioning/ - Pickle, joblib, ONNX, model registries
   - rest-api-fastapi-flask/ - Creating ML APIs with FastAPI and Flask
   - batch-vs-online-serving/ - Offline batch predictions vs real-time inference

4. **monitoring-and-mlops/** - Maintaining models in production
   - data-drift/ - Detecting changes in input data distribution
   - model-drift/ - Monitoring model performance degradation
   - logging-and-alerting/ - Logging predictions, errors, and setting up alerts
   - ci-cd-for-ml/ - Continuous integration and deployment for ML systems

## Resources

### 📚 Books

- **"Designing Machine Learning Systems"** by Chip Huyen - Comprehensive production ML guide
- **"Machine Learning Engineering"** by Andriy Burkov - End-to-end ML systems
- **"Building Machine Learning Powered Applications"** by Emmanuel Ameisen - Practical ML development
- **"Introducing MLOps"** by Treveil et al. - MLOps fundamentals
- **"Machine Learning Design Patterns"** by Lakshmanan, Robinson, Munn - Google's ML patterns
- **"Reliable Machine Learning"** by Cathy Chen et al. - ML in production best practices
- **"Feature Engineering for Machine Learning"** by Alice Zheng - In-depth feature engineering

### 🎥 Videos

- **Made With ML MLOps Course** - Free comprehensive MLOps curriculum
- **Andrew Ng's MLOps Specialization** (Coursera) - Production ML systems
- **Full Stack Deep Learning** - Production deep learning best practices
- **FastAPI Tutorial** - Building production APIs
- **MLflow Tutorial Series** - Experiment tracking and model management
- **Weights & Biases YouTube** - MLOps tools and practices
- **Google Cloud AI Platform** - Deployment tutorials
- **AWS SageMaker Tutorials** - Cloud ML deployment
- **Evidently AI YouTube** - Model monitoring and data drift

### 🎧 Podcasts

- **MLOps Community Podcast** - Production ML practices and tools
- **Gradient Dissent** (W&B) - ML engineering and infrastructure
- **The TWIML AI Podcast** - Production ML episodes
- **Practical AI** - Real-world deployment stories
- **Data Engineering Podcast** - Data pipelines and infrastructure
- **Software Engineering Daily: ML Ops** - Engineering perspectives on ML

### 📄 Articles & Papers

**Foundational Papers:**
- **"Hidden Technical Debt in ML Systems"** (Google) - Challenges of production ML
- **"Rules of Machine Learning"** (Google) - Best practices from Google
- **"The ML Test Score"** - Framework for production readiness
- **"MLOps: Continuous delivery and automation pipelines in ML"** - MLOps overview

**Practical Guides:**
- **MLOps Principles** (ml-ops.org) - Community MLOps standards
- **AWS Well-Architected ML Lens** - Cloud ML best practices
- **Google Cloud ML Best Practices** - Production ML patterns
- **Microsoft Azure ML Documentation** - Deployment guides
- **FastAPI Documentation** - Modern API framework
- **Scikit-learn Pipeline Documentation** - Building reproducible workflows

**Blog Posts:**
- **Netflix Tech Blog** - Recommendation systems at scale
- **Uber Engineering Blog** - ML platform (Michelangelo)
- **Airbnb Engineering** - ML infrastructure
- **Spotify Engineering** - Personalization at scale
- **Meta AI Blog** - Production deep learning

### 🌐 HTML/Interactive Resources

- **MLflow** (mlflow.org) - Experiment tracking, model registry, deployment
- **Weights & Biases** (wandb.ai) - Experiment tracking and collaboration
- **Evidently AI** (evidentlyai.com) - ML monitoring and testing
- **DVC** (dvc.org) - Data version control for ML
- **BentoML** (bentoml.com) - Model serving framework
- **Kubeflow** - ML workflows on Kubernetes
- **Seldon** - Model deployment on Kubernetes
- **FastAPI** (fastapi.tiangolo.com) - Modern API framework
- **Great Expectations** (greatexpectations.io) - Data validation
- **Prefect** / **Airflow** - Workflow orchestration
- **Streamlit** (streamlit.io) - ML app framework
- **Gradio** (gradio.app) - ML model demos

---

**Previous Module:** [05-special-topics](../05-special-topics/) - Special Topics and Applications

## Production ML Resources

### Tools & Platforms

**Experiment Tracking:**
- MLflow, Weights & Biases, Neptune.ai, Comet.ml

**Model Deployment:**
- FastAPI, Flask, BentoML, Seldon, TorchServe, TensorFlow Serving

**MLOps Platforms:**
- Kubeflow, MLflow, Metaflow, Kedro, ZenML

**Cloud ML Services:**
- AWS SageMaker, Google Vertex AI, Azure ML, Databricks

**Monitoring:**
- Evidently AI, Arize AI, Fiddler, WhyLabs

**Data Validation:**
- Great Expectations, Pandera, Deepchecks

**Feature Stores:**
- Feast, Tecton, Hopsworks

---

🎓 **Congratulations!** You've completed the OpenMLAtlas learning path. You now have a comprehensive understanding of machine learning from mathematical foundations to production deployment. Keep practicing, stay current with research, and continue building projects!
