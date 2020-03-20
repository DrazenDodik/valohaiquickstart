import tensorflow as tf
import os
import json
import numpy
import argparse


def getArgs():
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser()
    # Define two arguments that it should parse
    parser.add_argument('--epochnum', type=int, default=5)
    parser.add_argument('--learningrate', type=float, default=0.001)

    # Now run the parser that will return us the arguments and their values and store in our variable args
    args = parser.parse_args()

    # Return the parsed arguments
    return args

# Call our newly created getArgs() function and store the parsed arguments in a variable args. We can later access the values through it, for example args.learningrate
args = getArgs()

# A function to write JSON to our output logs
# with the epoch number with the loss and accuracy from each run.
def logMetadata(epoch, logs):
    print()
    print(json.dumps({
        'epoch': epoch,
        'loss': str(logs['loss']),
        'acc': str(logs['accuracy']),
    }))

metadataCallback = tf.keras.callbacks.LambdaCallback(on_epoch_end=logMetadata)

# Get the output path from the Valohai machines environment variables
output_path = os.getenv('VH_OUTPUTS_DIR')

# Get the path to the folder where Valohai inputs are
input_path = os.getenv('VH_INPUTS_DIR')

# Get the file path of our MNIST dataset that we defined in our YAML
mnist_file_path = os.path.join(input_path, 'my-mnist-dataset/mnist.npz')

with numpy.load(mnist_file_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learningrate),
        loss=loss_fn,
        metrics=['accuracy'])

model.fit(x_train, y_train, epochs=args.epochnum, callbacks=[metadataCallback])

# Save our model to that the output as model.h5
# /valohai/outputs/model.h5
model.save(os.path.join(output_path, 'model.h5'))