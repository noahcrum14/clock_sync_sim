import numpy as np

# Polarization states
POL_DICT = {
    'h': np.array([1, 0]),
    'v': np.array([0, 1]),
    'd': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
    'a': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
}

# Physical constants
SIGMA = 0.05                   # Spectral bandwidth [THz]
C_FIBER = 0.20818920694444445  # Speed of light in fiber [m/ns]
SIGMA_JITTER = 0.05 # Jitter standard deviation [ns]

# Experiment parameters
DEFAULT_PARAMS = {
    'parties': {
        'alice': {
            'source_type': "spdc", # or 'coherent'
            'rep_rate': 10e6,
            'delay': 370,
            'offset': 0,
            'mu' : 1
        },
        'bob': {
            'source_type': "spdc", # or 'coherent'
            'rep_rate': 10e6,
            'delay': 0,
            'offset': 28,
            'mu' : 1
        }
    },
    'distance_alice_to_bob': 19_800,
    'processing_window': 100,
    'search_initial_low': 19_700,
    'search_initial_high': 19_900,
    'search_tolerance': 0.1,
    'averaging_runs': 1,  # Number of runs per evaluation
    'optimizer': "golden_section",  # or "golden_section"
    'max_iter': 100,
    'learning_rate': 0.1,
    'momentum': 0.9,
}