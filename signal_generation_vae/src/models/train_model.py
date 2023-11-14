"""
Building and training a VAE architecture

Author: Miele Lorenzo, Lomazzi Luca
Date: tbd

This script defines a Variational Autoencoder (VAE) architecture for training and using a VAE model for 
signal reconstruction and latent space visualization. The VAE model consists of an encoder and a decoder, 
and it is designed to process and generate time-series signals. It applies directly on the DOI: 10.1038/s41597-019-0208-1.

This file start with a pkl dataset reconstructed by a database starting from a 
http://openguidedwaves.de/downloads/ in the script build_dataset.
To run this script you have to download the database, preprocess it via build_dataset and then
run this script

Usage:
1. Create an instance of the VAE model with a specified input shape, latent dimension, KL weight, and learning rate.
2. Compile and train the VAE on your dataset using the `compile_and_train` method.

Attributes:
    - latent_dim (int): The dimension of the latent space.
    - encoder (tf.keras.Model): The encoder model.
    - decoder (tf.keras.Model): The decoder model.
    - kl_weight (float): The weight for the Kullback-Leibler (KL) divergence in the loss function.
    - learning_rate (float): The learning rate for optimization.

Methods:
    - build_encoder(input_shape, latent_dim): Build the encoder model.
    - sampling(args): Sample from the latent space.
    - build_decoder(input_shape): Build the decoder model.
    - train_step(data): Perform a single training step for the VAE.
    - compile_and_train(data, num_epochs, batch_size): Compile and train the VAE model.

Example:
    # Instantiate a VAE model
    input_shape = (784,)  # Example input shape
    vae = VAE(input_shape, latent_dim=10, kl_weight=0.5, learning_rate=0.005)

    # Compile and train the VAE
    vae.compile_and_train(train_data, num_epochs=50, batch_size=64)

"""
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VAE(keras.Model):
    """Variational Autoencoder (VAE) implementation.

    This class defines a Variational Autoencoder model using TensorFlow and Keras.
    The VAE consists of an encoder, a decoder, and a sampling layer for the latent space.

    Args:
        input_shape (tuple): The shape of the input data.
        latent_dim (int): The dimension of the latent space (default is 2).

    Attributes:
        latent_dim (int): The dimension of the latent space.
        encoder (tf.keras.Model): The encoder model.
        decoder (tf.keras.Model): The decoder model.

    Methods:
        build_encoder(input_shape, latent_dim): Build the encoder model.
        sampling(args): Sample from the latent space.
        build_decoder(input_shape): Build the decoder model.
        train_step
        compile_and_train(data, num_epochs, batch_size): Compile and train the VAE.

    Example:
        # Instantiate a VAE model
        input_shape = (784,)  # Example input shape
        vae = VAE(input_shape, latent_dim=10)
        vae.compile_and_train(train_data, num_epochs=50, batch_size=64)
    """
    def __init__(self, input_shape, kl_weight, learning_rate, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_shape, latent_dim)
        self.decoder = self.build_decoder(input_shape)
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self, input_shape, latent_dim):
        encoder_inputs = keras.Input(shape = input_shape)
        x = layers.Dense(128, activation = 'silu')(encoder_inputs)
        x = layers.Dense(64, activation = 'silu')(x)
        # x = layers.Flatten()(x)
        x = layers.Dense(16, activation="silu")(x)
        z_mean = layers.Dense(latent_dim, name = "z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name = "z_log_var")(x)
        z = self.sampling([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="ENC")
        return encoder

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim)) # mean is 0 and sigma = 1
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_decoder(self, input_shape):
        # shape accepts a tuple of dimension
        decoder_inputs = keras.Input(shape = (self.latent_dim,))
        x = layers.Dense(16, activation = "silu")(decoder_inputs)
        x = layers.Dense(64, activation="silu")(x)
        # x = layers.Reshape((input_shape, 64))(x)
        x = layers.Dense(128, activation = "silu")(x)
        decoder_outputs = layers.Dense(input_shape, activation = "sigmoid")(x)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name = "DEC")
        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.mse(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# START: data/processed
# END:   model
#          - weights.h5
#          - my_encoder.keras
#          - my_decoder.keras

file_Path = os.path.abspath(__file__)
DATA_PATH = r"..\..\..\data\processed\processed_data.pkl"
OUTPUT_PATH_WEIGHTS = r"..\..\..\models\weights\standard_weights.h5"
OUTPUT_PATH_MODEL_DATA = r"..\..\..\models\model_data\standard_model_data.pkl"
FIG_PATH = r"..\..\..\reports\figures"

input_path = os.path.abspath(os.path.join(file_Path, DATA_PATH))
weights_path = os.path.abspath(os.path.join(file_Path, OUTPUT_PATH_WEIGHTS))
model_data_path = os.path.abspath(os.path.join(file_Path, OUTPUT_PATH_MODEL_DATA))

if not os.path.exists(input_path):
        raise FileNotFoundError(f"The specified data folder does not exist: {input_path}")

# Open the database generated from the build_dataset script
with open(input_path, 'rb') as file:
    processed = pickle.load(file)

signals = processed['signals']
data_dim = processed['data_dim']
length_catch = processed['length_catch']
temperature_number = processed['temperature_number']
t = processed["t"]

DURATION = 0.00131 # total duration of the signal

# Selecting the hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 20
KL_WEIGHT = 0.5
LEARNING_RATE = 0.005

# Rescaling is now performed to normalize the input in [0,1] interval
tensor_signals = tf.constant(signals, dtype=tf.float32)

min_input = tf.reduce_min(tensor_signals)
max_input = tf.reduce_max(tensor_signals)
interval = tf.subtract(max_input, min_input)
normalized_signals = tf.divide(tf.subtract(tensor_signals, min_input), interval)
print("\nMaximum of the inputs:", max_input.numpy())
print("Minimum of the inputs:", min_input.numpy())

# Debugging only
min_res= tf.reduce_min(normalized_signals)
max_res = tf.reduce_max(normalized_signals)
print("Maximum of the rescaled inputs:", max_res.numpy())
print("Minimum of the rescaled inputs:", min_res.numpy())

# The Test and Validation dataset aren't required, the metric is defined in the next script
print("The length of the training dataset is:", data_dim)
dataset = tf.data.Dataset.from_tensor_slices(normalized_signals)

# Dividing the dataset into train and test
train_dataset = (dataset.shuffle(data_dim).batch(BATCH_SIZE))

# # Debugging only
# print("First 5 elements of the train set")
# for sample in train_dataset.take(5):
#     print("Example:", sample.numpy())

# # Qui puoi chiedere anche il label dopo
# print("First 5 element of the test set:")
# for sample in test_dataset.take(5):
#     print("Example:", sample.numpy())

## Creation of the VAE
vae = VAE(length_catch, KL_WEIGHT, LEARNING_RATE)

print("Encoder Summary:")
vae.encoder.summary()
print("\nDecoder Summary:")
vae.decoder.summary()

vae.compile(optimizer=keras.optimizers.Adam(learning_rate = LEARNING_RATE))
vae.fit(train_dataset, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# summary of the input 
print(vae.layers[0].input)
# Creating the latent vector and reconstructing the input
_, _, z = vae.encoder(normalized_signals[0:1])
x_rec = vae.decoder(z)
plt.plot(t, normalized_signals[0,:], label = "Input signal")
plt.plot(t, x_rec[0,:], label = "Reconstructed signal")
plt.xlabel('t')
plt.ylabel('signal')
plt.legend()
plt.title('Input signal')
plt.show()

# Creating the 3D graph plotting with temperature labels
z_x = [] # first variable of z
z_y = [] # second variable of z
z_T = [] # Temperature relative of z
enc_dec_signals = [] # list of vectors
var_enc = [] # list of variance of the encoder

for i in range(normalized_signals.shape[0]):
    # the input require an interval of a single value
    z_mean, z_log_var, z = vae.encoder(normalized_signals[i:i+1])
    x_rec = vae.decoder(z_mean)
    z_x.append(z_mean[0,0])
    z_y.append(z_mean[0,1])
    enc_dec_signals.append(x_rec[0,:])
    var_enc.append(z_log_var)

z_T = [temp for temp in temperature_number]
    
# Plotting the latent space in 2D and saving the png
learning_rate_str = str(LEARNING_RATE).replace('.', '')
labels = [(int(temp) - int(temp) % 2) for temp in z_T]
fig1_name = f'\\2D_B{BATCH_SIZE}_LR{learning_rate_str}.png'
fig1_path = os.path.abspath(os.path.join(file_Path, FIG_PATH + fig1_name))
plt.figure(figsize=(12, 10))
plt.scatter(z_x, z_y, c=labels)
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.savefig(fig1_path, dpi=300)
plt.show()

# Plotting the latent space in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z_x, z_y, z_T, c=labels, marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Temperature')
plt.show()

fig2_name = f'\\3D_B{BATCH_SIZE}_LR{learning_rate_str}.png'
fig2_path = os.path.abspath(os.path.join(file_Path, FIG_PATH + fig2_name))
fig.savefig(fig2_path, dpi=300)

## Save the model if the input is linear and perform a linear regression 
ans = input("Is the latent space linear? (y/n): ")
if ans.lower() == 'y':
    
    # Check if the model already exist
    if os.path.exists(weights_path):
        ans = input("The model vae.weights already exist, want to overwrite it (y/n): ")
        if ans.lower() =='y':
            os.remove(weights_path)

    vae.save_weights(weights_path)

    # dictionary with the requested data
    data_to_save = {
        'z_x': z_x,
        'z_y': z_y,
        'z_T': z_T,
        't': t,
        'normalized_signals': normalized_signals,
        'length_catch': length_catch,
        'kl weight': KL_WEIGHT,
        'enc_dec_signals': enc_dec_signals,
        'log variance': var_enc,
        'BATCH_SIZE':BATCH_SIZE,
        'LEARNING_RATE':LEARNING_RATE
    }

    if os.path.exists(model_data_path):
        os.remove(model_data_path)

    # Save the dictionary in another file
    with open(model_data_path, 'wb') as file:
        pickle.dump(data_to_save, file)