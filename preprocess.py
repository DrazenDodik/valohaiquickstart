import os
import numpy as np

inputs_path = os.getenv('VH_INPUTS_DIR', './inputs')
outputs_path = os.getenv('VH_OUTPUTS_DIR', './outputs')

# Get path to raw MNIST dataset
input_path = os.path.join(inputs_path, 'my-raw-mnist-dataset/mnist.npz')

with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# Preprocess dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Output the preprocessed file
processed_file_path = os.path.join(outputs_path, 'mnist.npz')

np.savez(processed_file_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)