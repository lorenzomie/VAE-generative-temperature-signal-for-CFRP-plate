"""
Description:
This script demonstrates the utilization of the trained Variational Autoencoder (VAE)
to reconstruct signals in a latent space. The VAE was trained on Lamb wave signals
using TensorFlow and Keras.

Author: Lorenzo Miele, Lomazzi Luca

Usage:
- Make sure to have a trained VAE model with saved weights (vae.weights.h5).
- Ensure the existence of the 'standard_model_data.pkl' file containing the necessary data for
  the encoded space.

Note:
This script assumes the existence of 'standard_model_data.pkl', this file will be available
once a model is present in the folder models. Train model will generate this file if
the latent space is linear
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Writing the model class VAE as in the previous script

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

# Reconstruct the VAE model and retrieve the necessary data 

# Retrieve the data of the encoded space
if os.path.exists('z_data.pkl'):
    with open('z_data2.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
        z_x = loaded_data['z_x']
        z_y = loaded_data['z_y']
        z_T = loaded_data['z_T']
        t = loaded_data['t']
        normalized_signals = loaded_data['normalized_signals']
        length_catch = loaded_data['length_catch']
        KL_WEIGHT = loaded_data['kl weight']
        enc_dec_signals = loaded_data['enc_dec_signals']
        log_var = loaded_data['log variance']
        BATCH_SIZE = loaded_data['BATCH_SIZE']
        LEARNING_RATE = loaded_data['LEARNING_RATE']
        band_normalized_signals = loaded_data['Band normalized signals']
        band_temperature = loaded_data['Band temperature']
else:
    print(f"The file'{'z_data.pkl'}' doesn't exist.")

# creating the class
vae = VAE(length_catch, KL_WEIGHT, LEARNING_RATE)
vae.load_weights("vae.weights.h5")

# Transform all the variables into numpy array and initializing the data
z_x = np.array(z_x, dtype=float)
z_y = np.array(z_y, dtype=float)
z_T = np.array(z_T, dtype=float)

labels = [(int(temp) - int(temp) % 2) for temp in z_T]
learning_rate_str = str(LEARNING_RATE).replace('.', '')
t_par = np.linspace(-20, 20, np.shape(z_T)[0])

# Implementing a "3D regression model" by a PCA utilizing Singular Value Decomposition

data = np.concatenate((z_x[:, np.newaxis], 
                       z_y[:, np.newaxis], 
                       z_T[:, np.newaxis]), 
                      axis=1)

data_mean = data.mean(axis=0)

uu, dd, vv = np.linalg.svd(data - data_mean)

# the grid stands between 20° and 60° with 40° as mean
grid = np.mgrid[-20:20:0.01][:, np.newaxis]

line = vv[0] * grid

# shift by the mean to get the line in the right place
line += data_mean

# Now vv[0] contains the first principal component, i.e. the direction
# vector of the 'best fit' line in the least squares sense. vv[1] and vv[2] the variance in the other 2 dimensions
# 
# It's a straight line, so we only need 2 points.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter3D(z_x, z_y, z_T, c=labels, marker = 'o')
ax.plot3D(*line.T)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Temperature')
colorbar = plt.colorbar(scatter)
colorbar.set_label('Temperature Intensity')
plt.show()


# Selecting now the temperature it's possible to generate the signal 

def find_nearest(vector, value, dimension):
    
    idx = (np.abs(vector[:, dimension] - value)).argmin()
    return vector[idx]

# SELECT THE TEMPERATURE
TEMPERATURE = 23.70 #[°]
z1, z2, T_check = find_nearest(line, TEMPERATURE, dimension = 2)
print(f'Generating the signal with respect to {T_check:.2f}°')
print(f'The latent space point is: [{z1:.6f} , {z2: .6f}]\n')

# Retrieving the input signal with the nearest temperature

idx = (np.abs(z_T - TEMPERATURE)).argmin()
signal_val = normalized_signals[idx]
idx2 = (np.abs(np.array(band_temperature) - TEMPERATURE)).argmin()
training_temp_val = band_temperature[idx2]
training_signal_val = band_normalized_signals[idx2]
print(f'The nearest known signal is at T: {z_T[idx]:.2f}°')
print(f'The nearest training signal is at T: {training_temp_val:.2f}°')

# Utilizing now the vae model implemented in the previous script to reconstruct the signal

latent_vec = tf.constant([z1, z2], dtype=tf.float32)
latent_vec = tf.expand_dims(latent_vec, 0) # expand the input to have compatible dimensions
recon_signal = vae.decoder(latent_vec)

plt.plot(t, signal_val, label = f"Nearest signal in the dataset at T: {z_T[idx]:.2f}")
plt.plot(t, recon_signal[0], label = "Reconstructed signal")
plt.xlabel('t [s]')
plt.ylabel('signal')
plt.legend()
plt.title(f'Lamb Wave at {T_check:.2f}')
plt.show()
