"""Constants module."""

# Authors: Hamza Cherkaoui

import os
import numpy as np


# plotting options
FIGURES_DIR = '_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)
FONTSIZE = 18                                      # figure fontsize
LW = 5                                             # curves width
ALPHA = 0.4                                        # curves transarency

# launcher options
N_JOBS = 7  # 40
N_RUNS = 1 * N_JOBS                                # to limit the number of serial pass
SEED = 4578902                                     # reproducibility of the randomness

# list of parameters
L_EPS = np.logspace(start=-3, stop=3, num=50)      # default 'task difference' between target and source list
L_SIGMA_S = np.logspace(start=-3, stop=3, num=50)  # default source noise variance list

# data parameter
N_TRAIN_SAMPLES = 2000                             # global training samples to take or generate
N_T = 1000                                         # default number of target samples
STEP = 20                                          # default number consider when using logspace or linspace
SIGMA = 1.0                                        # default (target and source) noise variance
D = 50                                             # default dimension of the problem
EPS = 0.5                                          # default 'task difference' between target and source
LBDA_DEFAULT = 1.0                                 # default ridge regularization
