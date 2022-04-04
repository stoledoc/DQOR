# DQOR
Grading Diabetic Retinopathy and Prostate Cancer Diagnostic Images with Deep Quantum Ordinal Regression

![dqor](https://github.com/stoledoc/Resources/blob/master/dqor/dqor_model.png)

Implementation of Deep Quantum Ordinal Regressor

* Santiago Toledo-Cortés, Diego H. Useche, Henning Müller, and Fabio A. González.
## Abstract

Although for many diseases there is a progressive diagnosis scale, automatic analysis of grade-based medical images is quite often addressed as a binary classification problem, missing the finer distinction and intrinsic relation between the different possible stages or grades. Ordinal regression (or classification) considers the order of the values of the categorical labels and thus takes into account the order of grading scales used to assess the severity of different medical conditions. This paper presents a quantum-inspired deep probabilistic learning ordinal regression model for medical image diagnosis that takes advantage of the representational power of deep learning and the intrinsic ordinal information of disease stages. The method is evaluated on two different medical image analysis tasks: prostate cancer diagnosis and diabetic retinopathy grade estimation on eye fundus images. The experimental results show that the proposed method not only improves the diagnosis performance on the two tasks but also the interpretability of the results by quantifying the uncertainty of the predictions in comparison to conventional deep classification and regression architectures.

## Requirements

Python requirements:

- Tensorflow >= 2.3

## Download and Preprocessing of TCGA-PRAD

You can download the preprocessed TCGA-PRAD dataset in the following link:

* https://drive.google.com/drive/folders/14pbie6QsN64i0ArpfOnpiyEtFDX8LWqS?usp=sharing

## Download and Preprocessing of EyePACS and Messidor-2

Download EyePACS zip files from https://www.kaggle.com/c/diabetic-retinopathy-detection/data into `./data/eyepacs`. Run `$ ./eyepacs.sh` to decompress and preprocess the EyePACS data set, and redistribute it into a training and test set with gradable images. Run `$ ./messidor2.sh` to download, unpack, and preprocess the Messidor-2 data set.

## Quantum Measurement

Base code for Quantum Measurement Regressor can be found in the next repository:

* https://github.com/fagonzalezo/qmc

## Models

Pre-trained feature extraction models for binary prostate cancer diagnosis and binary diabetic retinopathy diagnosis can be found respectively in: 

 * https://github.com/juselara1/MLSA
 * https://github.com/stoledoc/DLGP-DR-Diagnosis
 
DQOR models for prostate cancer and diabetic retinopathy grading are stored in `./models`.


