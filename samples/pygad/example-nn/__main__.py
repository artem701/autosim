import numpy
import pygad.pygad.nn

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[1, 1],
                           [1, 0],
                           [0, 1],
                           [0, 0]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([0,
                            1,
                            1,
                            0])

# The number of inputs (i.e. feature vector length) per sample
num_inputs = data_inputs.shape[1]
# Number of outputs per sample
num_outputs = 2

HL1_neurons = 2

# Building the network architecture.
input_layer = pygad.pygad.nn.InputLayer(num_inputs)
hidden_layer1 = pygad.pygad.nn.DenseLayer(
    num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="relu")
output_layer = pygad.pygad.nn.DenseLayer(
    num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="softmax")

# Training the network.
pygad.pygad.nn.train(num_epochs=100,
               last_layer=output_layer,
               data_inputs=data_inputs,
               data_outputs=data_outputs,
               learning_rate=0.01)

# Using the trained network for predictions.
predictions = pygad.pygad.nn.predict(
    last_layer=output_layer, data_inputs=data_inputs)

# Calculating some statistics
num_wrong = numpy.where(predictions != data_outputs)[0]
num_correct = data_outputs.size - num_wrong.size
accuracy = 100 * (num_correct/data_outputs.size)
print(f"Number of correct classifications : {num_correct}.")
print(f"Number of wrong classifications : {num_wrong.size}.")
print(f"Classification accuracy : {accuracy}.")
