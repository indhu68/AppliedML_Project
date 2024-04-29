# Applied Machine Learning Project: Deep Learning Applications in Vehicular Wireless Communication

## Overview

This project explores the application of machine learning and deep learning techniques to vehicular wireless communication systems, focusing on Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication.

Vehicular communication networks face several challenges, including intracell interference, varying transmission distances, signal attenuation, and stringent communication latency requirements. V2I communication requires high data transmission capacity, while V2V communication places strong emphasis on maintaining low latency and reliable message delivery.

This project investigates machine learning and deep learning approaches to address these challenges through signal modeling, communication performance prediction, and learned communication strategies.

The primary objective is to explore approaches for optimizing communication performance by considering spectrum reuse, transmission power allocation, channel conditions, and message transmission reliability.

## Problem Statement

In hybrid V2V and V2I communication systems, wireless resources must be allocated efficiently to support different communication requirements.

The major challenges considered include:

- **Intracell Interference:** Interference between vehicles sharing wireless communication resources can affect signal quality and transmission reliability.
- **Transmission Distance:** Communication distance influences path loss and received signal strength.
- **Channel Variations:** Shadowing and noise introduce variations in signal-to-noise ratio (SNR).
- **Power Allocation:** Transmission power must be considered when optimizing communication performance.
- **Communication Reliability:** Messages must be transmitted and decoded reliably, particularly in latency-sensitive V2V communication.
- **Capacity and Latency Requirements:** V2I links prioritize communication capacity, whereas V2V links require low-latency communication.

The project explores a joint spectrum reuse and power allocation problem by decomposing the optimization problem into classification and regression tasks, alongside deep learning-based communication approaches.

## Project Objectives

- Explore machine learning techniques for vehicular wireless communication.
- Model wireless channel conditions using transmission distance, path loss, transmission power, noise, and shadowing.
- Calculate and analyze SNR under different communication scenarios.
- Investigate classification and regression models for communication-related prediction tasks.
- Develop recurrent neural network models for sequential signal-related learning.
- Implement autoencoder-based communication models for learning message representations and decoding transmitted information.
- Compare different autoencoder training approaches under varying channel conditions.
- Evaluate communication performance using metrics such as accuracy, Block Error Rate (BLER), and Root Mean Squared Error (RMSE).
- Study the effects of transmission power and the number of channel uses on communication performance.

## Methodology

### 1. Wireless Channel and SNR Modeling

The project models wireless communication conditions using parameters such as:

- Transmission distance
- Transmission power
- Noise floor
- Path Loss Exponent (PLE)
- Wavelength
- Shadowing variation
- Number of channel uses per message

The Signal-to-Noise Ratio (SNR) is calculated using a channel model that considers path loss and shadowing effects.

Different communication scenarios are generated to represent near, midrange, and far-distance communication, along with variations in transmission power and noise conditions.

These simulated channel conditions are used as inputs for machine learning and deep learning experiments.

### 2. Machine Learning-Based Classification and Regression

Machine learning models are explored to investigate the relationship between wireless channel conditions and communication characteristics.

**Classification**

The classification experiments use generated SNR values associated with different communication scenarios.

The scenarios include:

- Near-distance communication with high transmission power.
- Midrange communication with medium transmission power.
- Far-distance communication with low transmission power.
- High-power and noisy communication conditions.
- High-attenuation and distant communication conditions.

A Random Forest Classifier is used to learn patterns from the generated data and classify the corresponding communication scenarios.

**Regression**

Regression experiments investigate the prediction of communication distance using SNR-related information.

A Random Forest Regressor is used to model the relationship between SNR and transmission distance.

The experiments also consider different numbers of channel uses per message to examine their influence on the generated channel data.

**Model Development and Evaluation**

The machine learning workflow includes:

- Generating simulated communication datasets.
- Preparing input features and target labels.
- Splitting data into training and testing sets.
- Applying feature scaling using StandardScaler.
- Training classification and regression models.
- Exploring hyperparameter optimization using GridSearchCV.
- Evaluating predictions using classification reports, Mean Squared Error (MSE), and R² score.

### 3. Autoencoder-Based Wireless Communication

A major component of this project investigates deep learning-based autoencoder architectures for wireless communication.

The autoencoder learns an end-to-end communication representation consisting of:

- **Encoder:** Maps input messages into learned signal representations.
- **Wireless Channel:** Introduces channel effects such as noise and varying SNR.
- **Decoder:** Attempts to reconstruct the original transmitted message from the received signal.

The encoder performs average power normalization, and the decoder reconstructs the transmitted information after the channel effects.

The models are implemented using TensorFlow and TensorFlow's compatibility API.

**Autoencoder Approaches**

The experiments investigate three autoencoder variants:

1. **Unweighted Autoencoder (AE):** Uses a sparse softmax cross-entropy objective for message reconstruction.
2. **Weighted Autoencoder (AE_Weighted):** Uses a weighted cross-entropy objective to investigate the effect of message-dependent weighting during training.
3. **Traditional Baseline (AE_Tb):** Provides a comparison approach using the standard sparse softmax cross-entropy objective without the additional weighting formulation.

These approaches are trained and evaluated under different channel conditions to study their communication performance.

### 4. Recurrent Neural Network for Communication Modeling

The project also explores recurrent neural networks using Long Short-Term Memory (LSTM) layers.

A custom `SemanticRNN` model is implemented using TensorFlow Keras.

The architecture includes:

- Multiple LSTM layers.
- Dropout layers for regularization.
- Recurrent dropout within LSTM layers.
- A dense output layer for regression.
- A custom loss function.
- Adam optimizer for model training.

The model uses SNR-related data represented as tensors and investigates regression-based learning.

Training and validation datasets are created using TensorFlow's data pipeline. The experiments track training and validation RMSE to examine prediction performance.

### 5. Experimental Configurations

The project investigates different experimental configurations to study the effect of transmission power and channel representation.

The experiments include:

- **100 mW transmission power:** Autoencoder experiments under the corresponding transmission power configuration.
- **200 mW transmission power:** Experiments examining autoencoder performance under a higher transmission power configuration.
- **Different channel representations:** Configurations using four and eight real channel uses per message, corresponding to different complex-symbol representations.
- **Complex symbols:** Experiments involving complex-valued communication symbol representations.
- **Variable channel conditions:** Evaluation under varying and constant SNR conditions.

These experiments allow different model configurations and communication approaches to be examined under simulated wireless channel conditions.

## Performance Evaluation

The project uses multiple metrics to evaluate the machine learning models and communication systems.

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified or decoded messages. |
| Block Error Rate (BLER) | Measures the proportion of message blocks that are incorrectly decoded. |
| Root Mean Squared Error (RMSE) | Measures the prediction or reconstruction error. |
| Mean Squared Error (MSE) | Measures the average squared difference between predicted and actual values. |
| R² Score | Evaluates the goodness of fit of regression models. |

For autoencoder experiments, RMSE and BLER are examined across different channel conditions and communication distances.

The experiments also include constellation plotting to visualize learned communication symbol representations.

## Technologies and Frameworks

- **Python:** Core programming language.
- **NumPy:** Numerical computation and data generation.
- **Matplotlib:** Visualization of experimental results.
- **Scikit-learn:** Machine learning models, preprocessing, data splitting, and evaluation.
- **TensorFlow:** Deep learning model implementation and training.
- **TensorFlow Keras:** LSTM-based recurrent neural network implementation.
- **Jupyter Notebook / Google Colab:** Experimentation and development environments.

## Key Learning Outcomes

Through this project, I explored:

- Applying machine learning to wireless communication problems.
- Modeling wireless channel conditions and calculating SNR.
- Understanding the effects of path loss, shadowing, noise, and transmission distance.
- Developing classification and regression models for communication-related tasks.
- Implementing Random Forest models and hyperparameter tuning.
- Understanding encoder-channel-decoder architectures.
- Implementing autoencoder-based communication systems.
- Exploring weighted and unweighted training objectives.
- Building LSTM-based recurrent neural networks using TensorFlow.
- Evaluating communication reliability using BLER and reconstruction performance using RMSE.
- Investigating the relationship between transmission power, channel conditions, and learned communication representations.

## Conclusion

This project explores the application of machine learning and deep learning to challenges in vehicular wireless communication. By combining simulated wireless channel modeling, traditional machine learning algorithms, recurrent neural networks, and autoencoder-based communication architectures, it investigates different approaches to modeling and improving communication performance.

The experiments provide practical exposure to the intersection of artificial intelligence, deep learning, and wireless communication systems, particularly in the context of V2V and V2I networks.

## Project Repository

GitHub: https://github.com/indhu68/AppliedML_Project

