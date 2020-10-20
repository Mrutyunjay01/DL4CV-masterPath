# import the packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    """
    Class to baby sit our training process.
    """

    def __init__(self, figPath, jsonPath=None, startAt=0):
        """
        :argument
        figpath: output path for the figure
        jsonPath: path to the json serialized file
        startAt: starting Epoch
        """
        
    pass
