# In-ear-EEG
This code was written to validate the performance of a novel in-ear EEG device. The benchmarking process consisted of two phases: EEG Paradigm testing and Seizure detection.

## EEG Paradigms Analysis
4 Paradigms are used to evaluate the performance of the device: Alpha Modulation, Auditory Oddball, ASSR and SSVEP.  
  
Using the paradigm_testing.py file, and the in-ear devices connected to an OpenBCI board, the paradigms are reproduced and the raw data is collected and stored as a .csv file with a session name as defined by the user in the python file. Each subject performs the tests three times and the sessions are averaged using the averging.m file in which the three session names are stated at the start- ensuring the session names matched the ones defined in the python file.   
  
With several subjects performing the tests, the statistical_testing.m file performs grand average calculations for each paradigm test, calculating its statistical significance.

## Seizure Detection
The Zenodo neonatal EEG data set is used to train and test three seizure detection algorithms and assess their performance. Only patients with right hemishpere or bilateral seizures have been taken into account to develop an algorithm that can detect seizures using temporal data (as would be collected by an in-ear EEG device) regardless of seizure foci. The Zenodo data has been extracted using the ZDF browser and the data stored by patient number. With the data in the same directory as the seziure_algorithms.m file, the file can be run to train and test the algorithms.


