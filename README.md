# In-ear-EEG
This code was written to validate the performance of a novel in-ear EEG device. The benchmarking process consisted of two phases: EEG Paradigm testing and Seizure detection.

## EEG Paradigms Analysis
4 Paradigms are used to evaluate the performance of the device: Alpha Modulation, Auditory Oddball, ASSR and SSVEP.  
  
Using the paradigm_testing.py file, and the in-ear devices connected to an OpenBCI board, the paradigms are reproduced and the raw data is collected and stored as a .csv file with a session name as defined by the user in the python file. Each subject performs the tests three times and the sessions are averaged using the averging.m file in which the three session names are stated at the start- ensuring the session names matched the ones defined in the python file.   
  
With several subjects performing the tests, the statistical_testing.m file performs grand average calculations for each paradigm test, calculating its statistical significance.

## Seizure Detection
The Zenodo neonatal EEG data set is used to train and test three seizure detection algorithms and assess their performance using the seizure_algorithms.mat file. Only patients with right hemishpere or bilateral seizures have been taken into account to develop an algorithm that can detect seizures using temporal data (as would be collected by an in-ear EEG device) regardless of seizure foci. The Zenodo data has been extracted using the ZDF browser and the data stored by patient number in the same directory as the. With the data in the same directory as the seziure_algorithms.m file, the file can be run to train and test the algorithms.
### Feature Extraction
The data is sectioned into 5 second windows with 50% overlap and paired with the annotator’s data to generate a label ‘seizure’ if the expert annotated the presence of a seizure in more than 60% of the window. Using 5 levels of decomposition with the Daubechies wavelet, the mean, variance, kurtosis, and skew are extracted from each frequency band. This yields a total of 70044 feature vectors of size 1 x 20, which are normalised using z-score. A balanced data subset containing an equal number of seizure and non-seizure epochs is used to train the algorithms.
### The classifiers
1) SVM-based classifier with a Gaussian kernel
2) K-nearest neighbours (KNN) classifier with k=5
3) Fully connected neural network (FNN) with a hidden layer of 20 neurons was set up, utilising scale conjugate gradient 
backpropagation and a decision threshold of 0.5.


