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
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))*self.kl_weight
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

def load_processed_data(file_Path, DATA_PATH):
    input_path = os.path.abspath(os.path.join(file_Path, DATA_PATH))

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

    return signals, data_dim, length_catch, temperature_number, t

def normalize(signals):
    """
    Normalize a list of signals to the [0, 1] interval.

    Args:
        signals (list): List of input signals.

    Returns:
        tf.Tensor: Normalized signals.
    """
    tensor_signals = tf.constant(signals, dtype=tf.float32)
    
    min_input = tf.reduce_min(tensor_signals)
    max_input = tf.reduce_max(tensor_signals)
    interval = tf.subtract(max_input, min_input)
    normalized_signals = tf.divide(tf.subtract(tensor_signals, min_input), interval)
    
    print("\nMaximum of the inputs:", max_input.numpy())
    print("Minimum of the inputs:", min_input.numpy())
    print("\n")
    
    return normalized_signals

def display_model_options():
    print("\033[91mSELECT THE MODEL:\033[0m\n")
    print("1. Standard Model")
    print("2. Band Model")
    print("3. Sparse Model")

def get_band_data(temp_min, temp_max, temperature_number, normalized_signals):
    counter = 0
    indexes = []
    band_temperature = []

    for temp in temperature_number:
        if temp > temp_min and temp < temp_max:
            band_temperature.append(temp)
            indexes.append(counter)
        counter += 1
            
    model_signal = tf.gather(normalized_signals, tf.constant(indexes))
    data_dim = len(indexes)
    return band_temperature, model_signal, data_dim

def get_sparse_data(temp_int, cluster_int, temperature_number, normalized_signals, start_temp):
    
    if temp_int >= cluster_int:
        raise ValueError("You can't have a inter-cluster distance greater \
            than a cluster range")
    
    sparse_temperature = []
    indexes = []
    start_intervals = list(range(start_temp, 61, cluster_int))
    new_intervals = []
    
    for i in range(len(start_intervals)):
        end_temp = start_intervals[i] + temp_int
        if end_temp >= 60:
            end_temp = 60
        new_intervals.append(start_intervals[i]) 
        new_intervals.append(end_temp)
        

    for i in range(0, len(new_intervals), 2):
        counter = 0
        for temp in temperature_number:
            if temp > new_intervals[i] and temp < new_intervals[i+1]:
                sparse_temperature.append(temp)
                indexes.append(counter)
            counter += 1
    
    model_signal = tf.gather(normalized_signals, tf.constant(indexes))
    data_dim = len(indexes)
    
    return sparse_temperature, model_signal, data_dim
            

def get_model_data(model_type, normalized_signals, data_dim):
    if model_type == "1":
        print("You selected the Standard Model.")
        model_name = 'standard'
        model_signal = normalized_signals
        # data_dim will remain the same
        band_temperature = []
        sparse_temperature = []
        
    elif model_type == "2":
        print("You selected the Band Model.")
        model_name = 'band'
        
        ## Creating the Dataset using only a band of temperature for a different scope
        # use MIN_TEMP and MAX_TEMP to modify the band
        MIN_TEMP = 30
        MAX_TEMP = 50
        band_temperature, model_signal, data_dim = get_band_data(MIN_TEMP, \
            MAX_TEMP, temperature_number, normalized_signals)
        sparse_temperature = []
        
    elif model_type == "3":
        print("You selected the Sparse Model.")
        model_name = 'sparse'
        TEMP_INTERVAL = 2
        CLUSTER_INTERVAL = 5
        sparse_temperature, model_signal, data_dim = get_sparse_data(TEMP_INTERVAL, \
            CLUSTER_INTERVAL, temperature_number, normalized_signals, start_temp = 20)
        band_temperature = []
        
    else:
        print("Invalid choice. Please enter a valid number (1, 2, or 3).")
    return model_signal, data_dim, model_name, band_temperature, sparse_temperature

def get_user_choice():
    """
    SELECT THE MODEL DATA:
    STANDARD MODEL: It will run the training with all the temperature spectrum [20:60]
    BAND MODEL: It will run the training with a band between MIN_TEMP and MAX_TEMP of temperature
    SPARSE MODEL: It will run the training with sparse band of temperature [20:22, 28:30, 45:47, ...]
    """
    return input("Enter the number corresponding to the desired model (1, 2, or 3): ")

def plot_signal_example(t, normalized_signals, x_rec):
    
    plt.plot(t, normalized_signals[0,:], label = "Input signal")
    plt.plot(t, x_rec[0,:], label = "Reconstructed signal")
    plt.xlabel('t')
    plt.ylabel('signal')
    plt.legend()
    plt.title('Input signal')
    plt.show()

def init_latent_space(normalized_signal, vae):
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
    return z_x, z_y, z_T, enc_dec_signals, var_enc

def plot_and_save_latent_space_2D(z_x, z_y, z_T, learning_rate, file_Path, model_name):
    learning_rate_str = str(learning_rate).replace('.', '')
    labels = [(int(temp) - int(temp) % 2) for temp in z_T]
    fig1_name = f'\\2D_B{BATCH_SIZE}_LR{learning_rate_str}_{model_name}.png'
    save_path = os.path.abspath(os.path.join(file_Path, FIG_PATH + fig1_name))
    plt.figure(figsize=(12, 10))
    plt.scatter(z_x, z_y, c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_and_save_latent_space_3D(z_x, z_y, z_T, learning_rate, file_Path, model_name):
    learning_rate_str = str(learning_rate).replace('.', '')
    labels = [(int(temp) - int(temp) % 2) for temp in z_T]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_x, z_y, z_T, c=labels, marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    plt.show()

    fig2_name = f'\\3D_B{BATCH_SIZE}_LR{learning_rate_str}_{model_name}.png'
    fig2_path = os.path.abspath(os.path.join(file_Path, FIG_PATH + fig2_name))
    fig.savefig(fig2_path, dpi=300)

def save_model(weights_path):
    if os.path.exists(weights_path):
        ans = input("The model weights.vae already exist, want to overwrite it (y/n): ")
        if ans.lower() =='y':
            os.remove(weights_path)

    vae.save_weights(weights_path)
    print("weights saved.")
    
def save_model_data(z_x, z_y, z_T, t, normalized_signals, model_signal, length_catch, KL_WEIGHT, \
    enc_dec_signals, var_enc, BATCH_SIZE, LEARNING_RATE, temperature_number, band_temperature, \
    sparse_temperature, model_data_path):
    # dictionary with the requested data
    data_to_save = {
        'z_x': z_x,
        'z_y': z_y,
        'z_T': z_T,
        't': t,
        'normalized_signals': normalized_signals,
        'model_signal': model_signal,
        'length_catch': length_catch,
        'kl weight': KL_WEIGHT,
        'enc_dec_signals': enc_dec_signals,
        'log variance': var_enc,
        'BATCH_SIZE':BATCH_SIZE,
        'LEARNING_RATE':LEARNING_RATE,
        'Temperature': temperature_number,
        'Band temperature': band_temperature,
        'Sparse temperature': sparse_temperature
    }

    if os.path.exists(model_data_path):
        ans = input("The model data already exist, want to overwrite it (y/n): ")
        if ans.lower() =='y':
            os.remove(model_data_path)
            # Save the dictionary in models/model_data
            with open(model_data_path, 'wb') as file:
                pickle.dump(data_to_save, file)
            print("Model data saved.")
    else:
        with open(model_data_path, 'wb') as file:
            pickle.dump(data_to_save, file)
        print("Model data saved.")
    
if __name__ == "__main__":

    ### INPUT and OUTPUT path ###
    # START: data/processed
    # END:   models/weights/name
    #          - weights_name.h5
    #        models/model_data
    #          - model_data_name.pkl

    file_Path = os.path.abspath(__file__)
    DATA_PATH = r"../../../data/processed/processed_data.pkl"
    OUTPUT_PATH_WEIGHTS = r"../../../models/weights"
    OUTPUT_PATH_MODEL_DATA = r"../../../models/model_data/model_data_"
    FIG_PATH = r"../../../reports/figures"

    signals, data_dim, length_catch, temperature_number, t = load_processed_data(file_Path, DATA_PATH)

    ### MODELS CONSTANTS AND HYPERPARAMETERS ###
    # Select the hyperparameters, the figures will have the Learning Rate and
    # the Batch_size in their name when saved

    DURATION = 0.00131 # total duration of the signal

    NUM_EPOCHS = 100
    BATCH_SIZE = 20
    KL_WEIGHT = 0.05
    LEARNING_RATE = 0.005
    
    normalized_signals = normalize(signals)
    display_model_options()
    model_type = get_user_choice()
    model_signal, data_dim, model_name, band_temperature, sparse_temperature = \
        get_model_data(model_type, normalized_signals, data_dim)

    # The Test and Validation dataset aren't required, the metric is defined in the next script
    print("The length of the training dataset is:", data_dim)
    dataset = tf.data.Dataset.from_tensor_slices(model_signal)

    train_dataset = (dataset.shuffle(data_dim).batch(BATCH_SIZE))

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
    # plot_signal_example(t, normalized_signals, x_rec)

    # Creating the 3D graph plotting with temperature labels
    z_x, z_y, z_T, enc_dec_signals, var_enc = init_latent_space(normalized_signals, vae)
        
    # Plotting the latent space in 2D and saving the png
    plot_and_save_latent_space_2D(z_x, z_y, z_T, LEARNING_RATE, file_Path, model_name)
    
    # Plotting the latent space in 3D
    plot_and_save_latent_space_3D(z_x, z_y, z_T, LEARNING_RATE, file_Path, model_name)

    ## Save the model if the input is linear and perform a linear regression 
    ans = input("Is the latent space linear? (y/n): ")
    if ans.lower() == 'y':
        
        output_path = OUTPUT_PATH_WEIGHTS + '\\' + model_name +'\\vae.weights.h5'
        model_output_path = OUTPUT_PATH_MODEL_DATA + model_name + '.pkl'
        weights_path = os.path.abspath(os.path.join(file_Path, output_path))
        model_data_path = os.path.abspath(os.path.join(file_Path, model_output_path))

        save_model(weights_path)
        
        save_model_data(z_x, z_y, z_T, t, normalized_signals, model_signal, length_catch, KL_WEIGHT, \
        enc_dec_signals, var_enc, BATCH_SIZE, LEARNING_RATE, temperature_number, band_temperature, \
        sparse_temperature, model_data_path)