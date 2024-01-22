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
from matplotlib.ticker import MaxNLocator


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

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

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

def display_model_options():
    print("\n\033[91mSELECT THE MODEL:\033[0m\n")
    print("1. Standard Model")
    print("2. Band Model")
    print("3. Sparse Model")

def get_user_choice():
    """
    SELECT THE MODEL DATA:
    STANDARD : trained on all the dataset
    BAND : trained on a specified band
    SPARSE : trained on a sparse cluster of point
    """
    return input("Enter the number corresponding to the desired model (1, 2, or 3): ")

def get_data_from_dict(path):
    
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
        z_x = loaded_data['z_x']
        z_y = loaded_data['z_y']
        z_T = loaded_data['z_T']
        t = loaded_data['t']
        normalized_signals = loaded_data['normalized_signals']
        model_signal = loaded_data['model_signal']
        length_catch = loaded_data['length_catch']
        KL_WEIGHT = loaded_data['kl weight']
        enc_dec_signals = loaded_data['enc_dec_signals']
        log_var = loaded_data['log variance']
        BATCH_SIZE = loaded_data['BATCH_SIZE']
        LEARNING_RATE = loaded_data['LEARNING_RATE']
        temperature = loaded_data['Temperature']
        band_temperature = loaded_data['Band temperature']
        sparse_temperature = loaded_data['Sparse temperature']

    return z_x, z_y, z_T, t, normalized_signals, model_signal, length_catch, \
        KL_WEIGHT, enc_dec_signals, log_var, BATCH_SIZE, LEARNING_RATE, \
        temperature, band_temperature, sparse_temperature

def get_model(file_Path, FORCED):
    
    FLAG = True
    
    while FLAG:
        display_model_options()
        model_type = get_user_choice()
        if FORCED:
            if model_type == '1':
                OUTPUT_PATH_WEIGHTS = r"../../../models/weights/forced/standard/vae.weights.h5"
                OUTPUT_PATH_MODEL_DATA = r"../../../models/model_data/forced_model_data_STANDARD.pkl"
                print("You selected the Standard Model.\n")
                model_name = 'Standard'
                
            elif model_type == '2':
                OUTPUT_PATH_WEIGHTS = r"../../../models/weights/forced/band/vae.weights.h5"
                OUTPUT_PATH_MODEL_DATA = r"../../../models/model_data/forced_model_data_BAND.pkl"
                print("You selected the Band Model.\n")
                model_name = 'Band'
                
            elif model_type == '3':
                OUTPUT_PATH_WEIGHTS = r"../../../models/weights/forced/sparse/vae.weights.h5"
                OUTPUT_PATH_MODEL_DATA = r"../../../models/model_data/forced_model_data_SPARSE.pkl"
                print("You selected the Sparse Model.\n")
                model_name = 'Sparse'
                
            else:
                print("Invalid choice. Please enter a valid number (1, 2, or 3).")
        
        else:
            if model_type == '1':
                OUTPUT_PATH_WEIGHTS = r"../../../models/weights/standard/vae.weights.h5"
                OUTPUT_PATH_MODEL_DATA = r"../../../models/model_data/model_data_STANDARD.pkl"
                print("You selected the Standard Model.\n")
                model_name = 'Standard'
                
            elif model_type == '2':
                OUTPUT_PATH_WEIGHTS = r"../../../models/weights/band/vae.weights.h5"
                OUTPUT_PATH_MODEL_DATA = r"../../../models/model_data/model_data_BAND.pkl"
                print("You selected the Band Model.\n")
                model_name = 'Band'
                
            elif model_type == '3':
                OUTPUT_PATH_WEIGHTS = r"../../../models/weights/sparse/vae.weights.h5"
                OUTPUT_PATH_MODEL_DATA = r"../../../models/model_data/model_data_SPARSE.pkl"
                print("You selected the Sparse Model.\n")
                model_name = 'Sparse'
                
            else:
                print("Invalid choice. Please enter a valid number (1, 2, or 3).")
        
        
        weights_path = os.path.abspath(os.path.join(file_Path, OUTPUT_PATH_WEIGHTS))
        model_data_path = os.path.abspath(os.path.join(file_Path, OUTPUT_PATH_MODEL_DATA))

        if os.path.exists(weights_path) and os.path.exists(model_data_path):
            print('print("Collecting the data ...")')
            FLAG = False
        else:
            print(f"The model {model_name} doesn't exist")
            if input("Do you want to select another model (y/n)?").lower() == 'y':
                pass
            else:
                raise KeyboardInterrupt

    return weights_path, model_data_path, model_name

def get_training_latent_space(model_signal, vae, temperature, \
        band_temperature, sparse_temperature, model_type):
    z_x = []
    z_y = []
    
    if model_type == 'Standard':
        z_T = temperature
    elif model_type == 'Band':
        z_T = band_temperature
    elif model_type == 'Sparse':
        z_T = sparse_temperature
    else:
        print("Invalid choice. Please enter a valid number (1, 2, or 3).")
        
    for i in range(np.shape(model_signal)[0]):
       z_mean, _, _ = vae.encoder(model_signal[i:i+1])
       z_x.append(z_mean[0,0])
       z_y.append(z_mean[0,1])
       
    return z_x, z_y, z_T

       
def simple_PCA(z_x, z_y, z_T):
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
    # vector of the 'best fit' line in the least squares sense. 
    # vv[1] and vv[2] the variance in the other 2 dimensions
    
    return line

def find_nearest(vector, value, dimension):
    
    idx = (np.abs(vector[:, dimension] - value)).argmin()
    return vector[idx]

def plot_latent_space(z_x, z_y, z_T, line):
    # PLOT the Latent Space
    labels = [(int(temp) - int(temp) % 2) for temp in z_T]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(z_x, z_y, z_T, c=labels, marker = 'o')
    ax.plot3D(*line.T)
    ax.set_xlabel('x', fontsize=22, labelpad=10)
    ax.set_ylabel('y', fontsize=22, labelpad=10)
    ax.set_zlabel('Temperature [°C]', rotation = -180, fontsize=22, labelpad=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major', labelsize=16)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Temperature [°C]', fontsize = 22)
    colorbar.ax.tick_params(labelsize=16)
    plt.show()

def plot_gen_signal(input_temperature, band_temperature, sparse_temperature, \
                    line, z_T, normalized_signals, model_name, vae, t, cut_idx):
    """Generate a signal at a fixed Temperature

    Args:
        input_temperature (int): temperature by the user
    """
    z1, z2, T_check = find_nearest(line, input_temperature, dimension = 2)
    print(f'Generating the signal with respect to {T_check:.2f}°C')
    print(f'The latent space point is: [{z1:.6f} , {z2: .6f}]\n')

    # Retrieving the input signal with the nearest temperature

    idx = (np.abs(z_T - input_temperature)).argmin()
    signal_val = normalized_signals[idx]
    train_temp = temperature
    print(f'The nearest known signal is at T: {z_T[idx]:.2f}°C')
    
    if model_name == 'Band':
        idx2 = (np.abs(np.array(band_temperature) - input_temperature)).argmin()
        # Nearest signal in the training dataset (if you want to plot)
        training_signal_val = model_signal[idx2]
        training_temp_val = band_temperature[idx2]
        train_temp = band_temperature
        print(f'The nearest training signal is at T: {training_temp_val:.2f}°C')
    elif model_name == 'Sparse':
        idx2 = (np.abs(np.array(sparse_temperature) - input_temperature)).argmin()
        # Nearest signal in the training dataset (if you want to plot)
        training_signal_val = model_signal[idx2]
        training_temp_val = sparse_temperature[idx2]
        train_temp = sparse_temperature
        print(f'The nearest training signal is at T: {training_temp_val:.2f}°C')
        pass
    
    latent_vec = tf.constant([z1, z2], dtype=tf.float32)
    latent_vec = tf.expand_dims(latent_vec, 0) # expand the input to have compatible dimensions
    recon_signal = vae.decoder(latent_vec)
    plt.plot(t, signal_val, label = f"Nearest signal in the dataset at T: {z_T[idx]:.2f}°C")
    plt.axvline(x=t[cut_idx], color='red', linestyle='--', label='Cutting Line for metric')
    plt.plot(t, recon_signal[0], label = "Reconstructed signal")
    plt.xlabel('t [s]')
    plt.ylabel('normalized signal [-]')
    plt.legend()
    plt.title(f'Lamb Wave at {T_check:.2f}°C')
    plt.show()
    return train_temp

def plot_gen_signal_v2(input_temperature, line, z_T, normalized_signals, \
                    model_name, vae, t):
    
    if (len(input_temperature) != 4):
        raise ValueError("Input temperature must have a length of 4.")
        
    T_MID = 40
    idx_mid = (np.abs(z_T - T_MID)).argmin()
    mid_signal = normalized_signals[idx_mid]
    mid_temp = temperature
    signals = []
    sig_temp = []
    recon_signals = []
    xmin = 0
    xmax = 0.0003
    
    for temp in input_temperature:
        z1, z2, _ = find_nearest(line, temp, dimension = 2)
    
        # Retrieving the input signal with the nearest temperature

        idx = (np.abs(z_T - temp)).argmin()
        signal_val = normalized_signals[idx]
        
        latent_vec = tf.constant([z1, z2], dtype=tf.float32)
        latent_vec = tf.expand_dims(latent_vec, 0) # expand the input to have compatible dimensions
        recon_signal = vae.decoder(latent_vec)
        signals.append(signal_val)
        sig_temp.append(z_T[idx])
        recon_signals.append(recon_signal[0])
        
    # Adjust the distance between the rows of the subgraphs
    plt.subplots_adjust(hspace=0.5)
    
    # First signal
    plt.subplot(2, 2, 1)
    plt.plot(t, signals[0], label=f"Nearest signal at T: {sig_temp[0]:.2f}°C")
    plt.plot(t, recon_signals[0], label="Reconstructed signal")
    plt.plot(t, mid_signal, label=f"Signal at 40°")
    plt.xlabel('t [s]')
    plt.ylabel('normalized signal [-]')
    plt.legend()
    plt.title(f"Signal at {input_temperature[0]}°C")
    plt.xlim([xmin, xmax])
    
    # Second signal
    plt.subplot(2, 2, 2)
    plt.plot(t, signals[1], label=f"Nearest signal at T: {sig_temp[1]:.2f}°C")
    plt.plot(t, recon_signals[1], label="Reconstructed signal")
    plt.plot(t, mid_signal, label=f"Signal at 40°C")
    plt.xlabel('t [s]')
    plt.ylabel('normalized signal [-]')
    plt.legend()
    plt.title(f"Signal at {input_temperature[1]}°C")
    plt.xlim([xmin, xmax])
    
    # Third signal
    plt.subplot(2, 2, 3)
    plt.plot(t, signals[2], label=f"Nearest signal at T: {sig_temp[2]:.2f}°C")
    plt.plot(t, recon_signals[2], label="Reconstructed signal")
    plt.plot(t, mid_signal, label=f"Signal at 40°C")
    plt.xlabel('t [s]')
    plt.ylabel('normalized signal[-]')
    plt.legend()
    plt.title(f"Signal at {input_temperature[2]}°C")
    plt.xlim([xmin, xmax])
    
    # Fourth signal
    plt.subplot(2, 2, 4)
    plt.plot(t, signals[3], label=f"Nearest signal at T: {sig_temp[3]:.2f}°C")
    plt.plot(t, recon_signals[3], label="Reconstructed signal")
    plt.plot(t, mid_signal, label=f"Signal at 40°C")
    plt.xlabel('t [s]')
    plt.ylabel('normalized signal [-]')
    plt.legend()
    plt.title(f"Signal at {input_temperature[3]}°C")
    plt.xlim([xmin, xmax])

    # plt.suptitle(f'Generated Signals with model {model_name}')
    plt.show()
    
def plot_max(normalized_signals, z_T , vae, t, model_name):
    recon_signals = vae(normalized_signals)
    idx_max_signal = np.array(tf.argmax(normalized_signals, axis = 1))
    idx_max_recon = np.array(tf.argmax(recon_signals, axis = 1))
    labels = [(int(temp) - int(temp) % 2) for temp in z_T]
    t_max_signal = []
    t_max_recon = []
    max_signal = []
    max_recon = []
    counter = 0
    
    for idx1, idx2 in zip(idx_max_signal, idx_max_recon):
        t_max_signal.append(t[idx1])
        t_max_recon.append(t[idx2])
        max_signal.append(normalized_signals[counter, idx1])
        max_recon.append(normalized_signals[counter, idx2])
        counter += 1
    
    # Model signal
    plt.subplot(2, 1, 1)
    plt.scatter(t_max_signal, max_signal, label=f"time of the maximum of the input signal", c=labels)
    plt.xlabel('t [s]')
    plt.ylabel('Maximum of the input_signal')
    plt.colorbar()
    plt.legend()

    # Reconstructed signal
    plt.subplot(2, 1, 2)
    plt.scatter(t_max_recon, max_recon, label=f"time of the maximum of reconstructed signal", c=labels)
    plt.xlabel('t [s]')
    plt.ylabel('Maximum of the reconstructed signal')
    plt.colorbar()
    plt.legend()
    plt.suptitle(f'Maximum magnitude and time in {model_name}')
    plt.show()
    
def error_metric(normalized_signal, vae, relevance_idx):
    recon_signals = vae(normalized_signal)
    rmse = tf.sqrt(tf.reduce_mean(
        tf.square(normalized_signal[:, :relevance_idx] - recon_signals[:, :relevance_idx])))
    print(f"RMSE: {rmse:.6f}")

def plot_rmse(normalized_signals, temperature, vae, relevance_idx, model_name):
    RMSE = []
    labels = [(int(temp) - int(temp) % 2) for temp in z_T]
    recon_signals = vae(normalized_signals)
    cut_signals = normalized_signals[:, :relevance_idx]
    cut_recon_signals = recon_signals[:, :relevance_idx]
    for i in range(len(cut_signals[:, 0])):
        err = tf.sqrt(tf.reduce_mean(tf.square(cut_signals[i] - cut_recon_signals[i])))
        RMSE.append(err)

    plt.scatter(temperature, RMSE, c = labels)
    plt.xlabel('T [°C]', fontsize=22)
    plt.ylabel('RMSE', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18)
    cb = plt.colorbar()
    cb.set_label('Temperature [°C]', fontsize = 22)
    cb.ax.tick_params(labelsize=22)
    # plt.title(f'Root Mean Square Error in {model_name}')
    plt.show()

if __name__ == '__main__':

    # If you want the forced model type FORCE == True otherwise is False by default
    FORCED = False
    
    file_Path = os.path.abspath(__file__)
    weights_path, model_data_path, model_name = get_model(file_Path, FORCED)
    
    z_x, z_y, z_T, t, normalized_signals, model_signal, length_catch, \
    KL_WEIGHT, enc_dec_signals, log_var, BATCH_SIZE, LEARNING_RATE, \
    temperature, band_temperature, sparse_temperature = get_data_from_dict(model_data_path)

    
    # creating the class
    vae = VAE(length_catch, KL_WEIGHT, LEARNING_RATE)
    vae.encoder.summary()
    vae.decoder.summary()
    
    vae.load_weights(weights_path, FORCED)
    
    z_x_train, z_y_train, z_T_train = get_training_latent_space(model_signal, vae, temperature, \
        band_temperature, sparse_temperature, model_name)
    
    # Transform all the variables into numpy array and initializing the data
    z_x_train = np.array(z_x_train, dtype=float)
    z_y_train = np.array(z_y_train, dtype=float)
    z_T_train = np.array(z_T_train, dtype=float)
    
    z_x = np.array(z_x, dtype=float)
    z_y = np.array(z_y, dtype=float)
    z_T = np.array(z_T, dtype=float)

    line = simple_PCA(z_x_train, z_y_train, z_T_train)
    plot_latent_space(z_x, z_y, z_T, line)

    # SELECT THE TEMPERATURE
    TEMPERATURE = 23.70 #[°]
    MAX_IDX = 4500
    TEMP_ARRAY = [23, 29, 42, 58] # This array must have 4 elements
    
    train_temp = plot_gen_signal(TEMPERATURE, band_temperature, sparse_temperature, \
                    line, z_T, normalized_signals, model_name, vae, t, MAX_IDX)
    plot_gen_signal_v2(TEMP_ARRAY, line, z_T, normalized_signals, \
                    model_name, vae, t)
    plot_max(normalized_signals, z_T , vae, t, model_name)
    error_metric(normalized_signals, vae, MAX_IDX)
    plot_rmse(normalized_signals, temperature, vae, MAX_IDX, model_name)