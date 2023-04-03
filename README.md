# A Federated Learning Approach for Authentication and User Identification based on Behavioral Biometrics

# Authors
- Rafael Veiga 
- Iago Medeiros
- Cristiano B. Both
- Denis Rosário
- Eduardo Cerqueira

# Abstract

A smartphone can collect behavioral data without requiring additional actions on the user's part and without the need for additional hardware. In an active or continuous user authentication process, information from integrated sensors, such as touch, and gyroscope, is used to monitor the user continuously. These sensors can capture behavioral (touch patterns, accelerometer) or physiological (fingerprint, face) data of the user naturally interacting with the device. However, transferring data from multiple users' mobile devices to a server is not recommended due to user data privacy concerns. This paper introduces an FL approach to define a user's biometric behavior pattern for continuous user identification and authentication. We also evaluate whether FL can be helpful in behavioral biometrics.
Evaluation results compare CNNs in different epochs using FL and a centralized method with low chances of wrong predictions in user identification by the gyroscope.

# Big picture

![alt text](https://github.com/VeigarGit/BiometricBehaviorFL/blob/main/big.png)

# DatasetBrainRun.ipynb

In this Jupyter project, we do a complete analysis of BrainRun to understand how they collect and distribute this data. The times to span each sample and how this influences the data frame. The code shows the many groups created from the number of timestamps. With this, we prepare the minimum values we need to prepare the data frame we will use for Federated Learning. 

# federated_run_biometric_behavior.py

The python file is for running our simulation federated learning. In this, we choose some Convolutional neural networks to evaluate and the metrics we search to identify users like the False positive rate and rejection rate. Thus, we made a code to run federated learning in his main strategy FedAVG. We use some CNNs from Sktime-dl to run users using gyroscope sensors as unique identifiers.

## Citation

If you publish work that uses our project, please cite as follows: 

```bibtex
@article{veigafederated,
  title={A Federated Learning Approach for Authentication and User Identification based on Behavioral Biometrics},
  author={Veiga, Rafael and Both, Cristiano B and Medeiros, Iago and Ros{\'a}rio, Denis and Cerqueira, Eduardo}
}
```
