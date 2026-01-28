import numpy as np

def get_lambda(epoch, max_epochs, gamma=10):
    """
    Calculates the adaptation parameter lambda (alpha) according to the paper schedule.
    It starts at 0 (no adaptation) and gradually increases to 1.
    
    Formula: 2 / (1 + exp(-gamma * p)) - 1, where p is progress (0 to 1).
    """
    p = epoch / max_epochs
    return 2. / (1. + np.exp(-gamma * p)) - 1.