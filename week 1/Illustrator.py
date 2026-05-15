import numpy as np
import matplotlib.pyplot as plt

class Illustrator:
    """
    A class to visualize neural activity data.
    Please finish the implementation and annotations of this class.
    """
    __slot__ = ['observation', 'trial_cnt', 'timestep_cnt', 'neuron_cnt']

    def __init__(self, observation: np.ndarray):
        """
        Initialize the Illustrator with the given observation data.
        Accepts a 3D numpy array of shape (Trials, Timepoints, Neurons) and stores it for visualization.
        Parameters:
            observation (np.ndarray): A 3D array containing neural activity data with dimensions (Trials, Timepoints, Neurons).
        """
        self.observation = observation
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = observation.shape