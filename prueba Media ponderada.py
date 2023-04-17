import tensorflow as tf
import numpy as np

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,), activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Define the loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Define the data and labels for the clients
client_data = [
    np.random.rand(100, 784),
    np.random.rand(100, 784),
    np.random.rand(100, 784),
    np.random.rand(100, 784)
]
client_labels = [
    np.random.randint(10, size=(100,)),
    np.random.randint(10, size=(100,)),
    np.random.randint(10, size=(100,)),
    np.random.randint(10, size=(100,))
]

# Define the training function for each client
def train_client(client_model, client_data, client_labels):
    client_model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    history = client_model.fit(client_data, client_labels, epochs=5, batch_size=10)
    return client_model, history.history['accuracy'][-1]

# Define the federated averaging function
def federated_averaging(client_models):
    new_weights = []
    for variables in zip(*[client_model.trainable_variables for client_model in client_models]):

        new_weight = tf.add_n(variables) / len(client_models)
        new_weights.append(new_weight)
    return new_weights

# Initialize the client models
client_models = [
    tf.keras.models.clone_model(model),
    tf.keras.models.clone_model(model),
    tf.keras.models.clone_model(model),
    tf.keras.models.clone_model(model)
]
accs = []
# Perform Federated Learning for 4 rounds
for round_num in range(100):
    print(f"Round {round_num+1}")
    # Train each client model on its data
    client_accuracies = []
    for i in range(len(client_models)):
        client_models[i], acc = train_client(client_models[i], client_data[i], client_labels[i])
        client_accuracies.append(acc)
    avg_accuracy = sum(client_accuracies) / len(client_accuracies)
    accs.append(avg_accuracy)
    print(f"Average accuracy of the client models in round {round_num+1}: {avg_accuracy}")
    # Compute the new global model weights using federated averaging
    new_weights = federated_averaging(client_models)
    # Set the new weights for the global model
    model.set_weights(new_weights)

print(accs)