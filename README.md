# fault_detection_alg
Fault detection for MEG-MRI. This program analyzes a set of magnetic field measurements from a MEG SQUID array and flags all faulty signals. Analysis consists of two main components:

- **Four different Fault Detection Filters (FDFs)**. Analyse signals individually and flag parts with errors. The user may choose which FDFs to run. The available FDFs are Uniq, Flat, Spike and Fourier.
- **Consistency analysis (CA)**. Compares signals from nearby detectors and flags ones which are not physically consistent. Optional.

Can be run with two modes: 

- **Mode 1: Full analysis**. Analyses entire signal. Will return bad segments (parts of signal with unambiguous errors) and suspicious segments (parts of signal containing possible errors).
- **Mode 2: Partial analysis**. Analyses user-defined time window from signals. Returns segments with no errors.

## How to run
Run ` main.py ` using Python from the terminal. Takes in the following command line arguments:

- `--filename` : *string*. Filename of the .npz file containing the data to analyse.
- ` -m `/` --mode ` : `1` or `2`. Determines which of the above modes to run. Required.
- ` -t`/` --time ` : two *floats*. Determines the time window for mode 2. Must be between 0.21 and 0.5.
- ` --filters ` : any combination of `uniq`, `flat`, `spike` and `fourier`. The names of the FDFs to run. 
- ` -p`/ `--physicality`. If this flag is present, CA is run.
- `-o`/ `--output`: *string*. Name of the file the results are written to.
- `--plot`. If this flag is present, the program plots the signals in order along with their results. To see the next signal, x out of the current plot. Segments in red are bad, segments in yellow are suspicious and segments in green are good (only in mode 2).
- `-prnt`/`--print_mode` : `print` or `file` or `none`. Determines where to print detailed information. `print` for terminal, `file` for log file and `none` for nowhere.
- `-log` : *string*. Filename to output detailed information to when `file` is chosen with the previous argument. Default is `log.log`
