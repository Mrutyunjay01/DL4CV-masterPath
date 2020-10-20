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
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

        pass

    def on_train_begin(self, logs=None):
        """
        Kicks off once when the training begins.
        :param logs:
        :return: None
        """
        if logs is None:
            logs = {}

        # initialize the history dictionary
        self.H = {}

        # if the json path exist, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check if a starting epoch is provided
                if self.startAt > 0:
                    # loop over the history log and trim any entires that are
                    # past the starting epoch
                    for key in self.H.keys():
                        self.H[key] = self.H[key][:self.startAt]

    def on_epoch_end(self, epoch, logs=None):
        """
        This function is called once a training epoch ends.
        :param epoch: completed last epoch
        :param logs:
        :return:
        """
        if logs is None:
            logs = {}

        # loop over the logs and update the loss, accuracy, etc
        # for the entire training process

        for (key, value) in logs.items():
            l = self.H.get(key, [])
            l.append(value)
            self.H[key] = l

        # check to see if the training history should be serialized to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        # construct the plot
        # ensure that at least two epochs have passed before plotting
        if len(self.H["loss"]) > 1:
            epochs_x = np.arange(0, len(self.H["loss"]))

            plt.style.use("ggplot")
            fig, axs = plt.subplots(1, 2)

            axs[0].plot(epochs_x, self.H["loss"], label="train_loss")
            axs[0].plot(epochs_x, self.H["val_loss"], label="val_loss")
            axs[0].set_xlabel("#Epochs")
            axs[0].set_ylabel("Loss")
            axs[0].set_title("Training/val loss")

            axs[0].plot(epochs_x, self.H["accuracy"], label="train_acc")
            axs[0].plot(epochs_x, self.H["val_accuracy"], label="val_acc")
            axs[0].set_xlabel("#Epochs")
            axs[0].set_ylabel("Accuracy")
            axs[0].set_title("Training/val Accuracy")

            fig.legend()
            fig.show()
            fig.savefig(self.figPath)
        pass

    pass
