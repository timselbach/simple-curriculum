# training/utils.py

import math

def competence_function(t, T, c0=0.01):
    """
    Computes the competence level at step t given total steps T and initial competence c0.
    """
    competence = (t * (1 - c0 ** 2) / T + c0 ** 2)**(1/2)
    return min(1.0, competence)
