# fault_detection_alg
Fault detection for MEG-MRI. This program analyzes a set of magnetic field measurements from a MEG SQUID array and flags all faulty signals. Analysis consists of two main components:

- **Four different Fault Detection Filters (FDFs)**. Analyse signals individually and flag parts with errors. Will always be run.
- **Consistency analysis (CA)**. Compares signals from nearby detectors and flags ones which are not physically consistent. Optional.

Can be run with two modes: 

- **Mode 1: Full analysis**. Analyses entire signal. Will return bad segments (parts of signal with unambiguous errors) and suspicious segments (parts of signal containing possible errors).
- **Mode 2: Partial analysis**. Analyses user-defined time window from signals. Returns segments with no errors.

## How to run
Run ` main.py ` using Python from the terminal. Takes in the following command line arguments:

- `--filename` : *string*. Filename of the .npz file containing the data to analyse.
- ` -m `/` --mode ` : `1` or `2`. Determines which of the above modes to run. Required.
