# In-ear-EEG
This code was written to validate the performance of a novel in-ear EEG device. The benchmarking process consisted of two phases: EEG Paradigm testing and Seizure detection.

## EEG Paradigms Analysis
4 Paradigms were used to evaluate the performance of the device: Alpha Modulation, Auditory Oddball, ASSR and SSVEP. Three recordings for each subject and each paradigm are averaged using the averging.m file in which the three session names as stated at the start. These files are raw EEG data in the format of .csv files extracted from an OpenBCI board using the paradigm_testing.py file.

## Seizure Detection
The three algorithms described take data from the Zenodo neonatal EEG dataset. Patients with right hemishpere or bilateral seizures were taken into account to develop an algorithm that can detect seizures from temporal data as would be collected by an in-ear EEG device. 


