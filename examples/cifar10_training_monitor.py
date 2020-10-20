import os
import argparse
import matplotlib

matplotlib.use("Agg")

from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer

from DL4CV.NeuralNetworks.miniVGGnet import MiniVGGNet
from DL4CV.CallBacks.trainingMonitor import TrainingMonitor

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="Path to the output Directory")
args = vars(ap.parse_args())

# show information on the Process ID
print(f"[INFO] Process ID: {os.getpid()}")

# load the training data and testing data
# scale them into range(0, 1)
print("[INFO] Loading the CIFAR-10 data...")
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

# conert the labels from integers to vectors
lb = LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
Y_test = lb.fit_transform(Y_test)

# initialize the label names of the cifar-10 dataset
labelNames = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

# initialize the SGD optimizer but without any learning rate decay
print("[INFO] Compiling the model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], f"{os.getpid()}.png"])
jsonPath = os.path.sep.join([args["output"], f"{os.getpid()}.json"])

callbacks = [TrainingMonitor(figPath=figPath, jsonPath=jsonPath)]

print("[INFO] Training the network...")
model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
