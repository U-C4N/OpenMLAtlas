# 🧠 04 - Deep Learning

Welcome to the **Deep Learning** module of OpenMLAtlas! This is where you'll dive into neural networks and modern AI. From basic perceptrons to transformers and large language models, you'll learn the techniques powering today's most exciting AI applications.

## 🎓 Overview

This module covers neural networks and deep learning—the technology behind computer vision, natural language processing, speech recognition, and generative AI. You'll learn both the theoretical foundations and practical implementation using PyTorch and TensorFlow/Keras.

## 🧩 Module Structure

| Directory | Topic | Description |
|-----------|-------|-------------|
| ⚡ `dl-foundations/` | **Deep Learning Foundations** | **Core neural network concepts** |
| ↳ `perceptron-and-mlp/` | Perceptron & MLP | Single neurons and multi-layer perceptrons |
| ↳ `backpropagation/` | Backpropagation | How neural networks learn via gradient descent |
| ↳ `initialization-and-activations/` | Initialization & Activations | Weight initialization, ReLU, sigmoid, tanh, etc. |
| ↳ `optimization-sgd-adam/` | Optimization | SGD, momentum, Adam, learning rate schedules |
| ↳ `regularization-dropout-batchnorm/` | Regularization | Dropout, batch normalization, early stopping |
| 🖼️ `computer-vision/` | **Computer Vision** | **Neural networks for images** |
| ↳ `cnn-basics/` | CNN Basics | Convolutions, pooling, feature maps |
| ↳ `modern-cnns-resnet-densenet/` | Modern CNNs | ResNet, DenseNet, EfficientNet architectures |
| ↳ `transfer-learning/` | Transfer Learning | Fine-tuning pre-trained models |
| 🔄 `sequence-modeling/` | **Sequence Modeling** | **Neural networks for sequential data** |
| ↳ `rnn-lstm-gru/` | RNN, LSTM, GRU | Recurrent architectures for sequences |
| ↳ `sequence-to-sequence/` | Seq2Seq Models | Encoder-decoder architectures |
| ↳ `attention-for-sequences/` | Attention Mechanisms | Attention for improved sequence modeling |
| 🤖 `transformers-and-llms/` | **Transformers & LLMs** | **Modern AI architecture** |
| ↳ `attention-basics/` | Attention Basics | Self-attention and multi-head attention |
| ↳ `transformer-architecture/` | Transformer Architecture | Encoder-decoder, positional encoding, layer norm |
| ↳ `llm-overview/` | Large Language Models | GPT, BERT, modern LLM architectures |
| 🔧 `frameworks/` | **Deep Learning Frameworks** | **Practical implementation tools** |
| 🔥 `frameworks/pytorch/` | **PyTorch** | **Primary DL framework** |
| ↳ `pytorch-basics/` | PyTorch Basics | Tensors, modules, basic workflows |
| ↳ `tensors-and-autograd/` | Tensors & Autograd | Automatic differentiation in PyTorch |
| ↳ `datasets-and-dataloaders/` | Data Loading | Dataset classes, DataLoader, augmentation |
| ↳ `training-loops/` | Training Loops | Complete training and validation pipelines |
| ↳ `pytorch-model-export-deployment/` | Export & Deployment | TorchScript, ONNX, model serving |
| 📊 `frameworks/tensorflow-keras/` | **TensorFlow & Keras** | **Alternative DL framework** |
| ↳ `tf-keras-basics/` | TF/Keras Basics | Sequential, Functional, Subclassing APIs |
| ↳ `tf-data-pipelines/` | TF Data Pipelines | tf.data for efficient data loading |
| ↳ `custom-training-loops/` | Custom Training Loops | GradientTape and custom training |
| ↳ `tf-serving-and-tf-lite/` | TF Serving & TF Lite | Model deployment and edge devices |

## 🗺️ Learning Path

We recommend following this order:

1. **dl-foundations/** - Start with the fundamentals
   - **perceptron-and-mlp/** - Understand basic neural networks
   - **backpropagation/** - Learn how networks learn
   - **initialization-and-activations/** - Master network setup
   - **optimization-sgd-adam/** - Learn training techniques
   - **regularization-dropout-batchnorm/** - Prevent overfitting
2. **frameworks/pytorch/** - Learn practical implementation
   - **pytorch-basics/** - Get started with PyTorch
   - **tensors-and-autograd/** - Understand automatic differentiation
   - **datasets-and-dataloaders/** - Handle data efficiently
   - **training-loops/** - Build complete training pipelines
   - **pytorch-model-export-deployment/** - Deploy your models
3. **computer-vision/** - Apply to images
   - **cnn-basics/** - Learn convolutional networks
   - **modern-cnns-resnet-densenet/** - Study state-of-the-art architectures
   - **transfer-learning/** - Leverage pre-trained models
4. **sequence-modeling/** - Apply to sequential data
   - **rnn-lstm-gru/** - Learn recurrent networks
   - **sequence-to-sequence/** - Build encoder-decoder models
   - **attention-for-sequences/** - Add attention mechanisms
5. **transformers-and-llms/** - Master modern AI
   - **attention-basics/** - Understand self-attention
   - **transformer-architecture/** - Learn the transformer
   - **llm-overview/** - Explore large language models
6. **frameworks/tensorflow-keras/** - (Optional) Learn alternative framework
   - **tf-keras-basics/** - Get started with TensorFlow
   - **tf-data-pipelines/** - Efficient data handling
   - **custom-training-loops/** - Advanced training control
   - **tf-serving-and-tf-lite/** - Production deployment

However, feel free to jump to specific topics based on your needs!

## 🔑 What You'll Learn

### ⚡ Neural Network Foundations
- **Network Architecture**: Perceptrons, MLPs, activation functions
- **Learning Process**: Backpropagation, gradient descent, optimization
- **Best Practices**: Initialization, normalization, regularization

### 🖼️ Computer Vision with Deep Learning
- **CNNs**: Convolutional layers, pooling, feature hierarchies
- **Modern Architectures**: ResNet, DenseNet, EfficientNet, Vision Transformers
- **Transfer Learning**: Fine-tuning pre-trained models for your tasks

### 🔄 Sequential Data Processing
- **RNNs**: Handling sequential and temporal data
- **Advanced RNNs**: LSTMs and GRUs for long-term dependencies
- **Seq2Seq**: Encoder-decoder models for translation, summarization

### 🤖 Transformers and Modern AI
- **Attention Mechanisms**: Self-attention and multi-head attention
- **Transformer Architecture**: The foundation of modern NLP and beyond
- **Large Language Models**: GPT, BERT, and state-of-the-art LLMs

### 🔧 Framework Mastery
- **PyTorch**: Industry-standard framework for research and production
- **TensorFlow/Keras**: Google's framework for production deployment
- **Model Deployment**: Exporting, serving, and optimizing models

## 📋 Prerequisites

Before starting this module, you should be familiar with:
- **Linear Algebra**: Matrix multiplication, gradients, chain rule
- **Calculus**: Partial derivatives, gradients, optimization
- **Python**: Object-oriented programming, NumPy
- **Classical ML**: Model training, evaluation, overfitting
- **Machine Learning Basics**: Loss functions, train/validation/test splits
- Completed **[03-classical-algorithms/](../03-classical-algorithms/)** or equivalent knowledge

## 🎬 Getting Started

1. Ensure you have completed the prerequisites
2. Install required libraries:
   ```bash
   # For PyTorch (CUDA 11.8 example - adjust based on your system)
   pip install torch torchvision torchaudio

   # For TensorFlow
   pip install tensorflow

   # Additional utilities
   pip install numpy pandas matplotlib jupyter
   ```
3. **GPU Setup** (highly recommended for deep learning):
   - Check if you have a CUDA-compatible GPU
   - Install appropriate CUDA toolkit and cuDNN
   - Verify GPU access in PyTorch: `torch.cuda.is_available()`
4. Start with `dl-foundations/` before moving to applications
5. Work through frameworks section to build practical skills

## 📚 How to Use This Module

Each subdirectory contains:
- **Theory**: Mathematical foundations and intuitive explanations (`.md` files)
- **Jupyter Notebooks**: Interactive implementations with visualizations (`.ipynb` files)
- **Code Examples**: Complete, runnable neural network implementations
- **Architecture Diagrams**: Visual representations of network structures
- **Exercises**: Build and train networks from scratch
- **Projects**: Real-world applications (image classification, text generation, etc.)
- **Resources**: Papers, blog posts, and additional references

### 🔧 Working with Jupyter Notebooks

To run the interactive examples:
```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab for a better experience
jupyter lab
```

Each notebook includes:
- 📝 Detailed explanations of network architectures
- 📊 Training curves, loss plots, and performance metrics
- 🧪 Experiments with different hyperparameters
- 🎨 Visualizations of learned features and attention patterns
- 💪 Hands-on exercises to build networks yourself
- 🎯 Real datasets (MNIST, CIFAR-10, text corpora, etc.)

### ⚡ GPU Acceleration

Deep learning benefits enormously from GPU acceleration. The notebooks will detect and use GPUs automatically. If you don't have a GPU:
- Use Google Colab (free GPU access)
- Use smaller models and datasets
- Reduce batch sizes and epochs

## 🚀 Next Steps

Once you've mastered deep learning fundamentals, move on to:
- **[05-special-topics/](../05-special-topics/)** - Domain-specific applications (NLP, generative models, RL, etc.)

## 💬 Contributing

Found an error? Have a suggestion? Feel free to open an issue or submit a pull request!

## 📜 License

This project is part of OpenMLAtlas - An open-source machine learning learning resource.

---

✨ **Remember**: Deep learning is powerful but requires substantial computational resources and data. Start with small experiments, understand the fundamentals, and gradually scale up. GPU access will make your learning journey much faster!
