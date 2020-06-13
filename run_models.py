#!/usr/bin/env python
import subprocess

# Control Run: lr=1e-3, epoch=25, bs=16, ap=2
file_extension = "-control"
plot_name = "plot" + file_extension + '.png'
model_name = "covid" + file_extension + '.png'
file_name = "output" + file_extension + '.txt'
subprocess.call(["python3", "train_model.py", "-d", "dataset", "-p", plot_name, "-m", model_name, "-o", file_name])

# Learning param: 1e-2, 1e-3, 1e-4
# Epochs: 15, 25, 35
# Batch Sizes: 8, 16, 32
# Average Pooling: 4x4, 2x2

# Learning Rate Testing
parameters = [1e-2, 1e-4]
for param in parameters:
    file_extension = "-lr-" + str(param)
    plot_name = "plot" + file_extension + '.png'
    model_name = "covid" + file_extension + '.model'
    file_name = "output" + file_extension + '.txt'
    subprocess.call(["python3", "train_model.py", "-d", "dataset", "-p", plot_name, "-m", model_name, "-o", file_name, "-r", str(param)])

# Epoch Testing
parameters = [15, 35]
for param in parameters:
    file_extension = "-epoch-" + str(param)
    plot_name = "plot" + file_extension + '.png'
    model_name = "covid" + file_extension + '.model'
    file_name = "output" + file_extension + '.txt'
    subprocess.call(["python3", "train_model.py", "-d", "dataset", "-p", plot_name, "-m", model_name, "-o", file_name, "-e", str(param)])

# Batch Size Testing
parameters = [8, 32]
for param in parameters:
    file_extension = "-bs-" + str(param)
    plot_name = "plot" + file_extension + '.png'
    model_name = "covid" + file_extension + '.model'
    file_name = "output" + file_extension + '.txt'
    subprocess.call(["python3", "train_model.py", "-d", "dataset", "-p", plot_name, "-m", model_name, "-o", file_name, "-b", str(param)])

# Average Pooling
parameters = [1, 3, 4]
for param in parameters:
    file_extension = "-ap-" + str(param)
    plot_name = "plot" + file_extension + '.png'
    model_name = "covid" + file_extension + '.model'
    file_name = "output" + file_extension + '.txt'
    subprocess.call(["python3", "train_model.py", "-d", "dataset", "-p", plot_name, "-m", model_name, "-o", file_name, "-a", str(param)])