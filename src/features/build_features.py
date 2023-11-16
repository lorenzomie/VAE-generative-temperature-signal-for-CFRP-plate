"""
Build feature vectors from the given database.

Author: Lorenzo Miele, Luca Lomazzi

Parameters:
-----------
database_path : str, optional
    File path to the database (default: 'database1.pkl').
    
frequencies : list of int, optional
    Frequencies (kHz) to include in feature vectors.
    If None, include all frequencies (default: None).
    
all_data : bool, optional
    Include all data regardless of frequencies (default: False).

Returns:
--------
signals : list of np.ndarray
    Feature vectors, each corresponding to a frequency.
    Flattened row vector representing the catch signal.

data_dim : int
    length of the data of signals
length_catch : int
    Length of the catch signal vectors.

temperature_numbers : list of float
    Mean temperature values corresponding to each feature vector.

Raises:
-------
FileNotFoundError
    If the specified database file is not found.

Dependencies:
-------------
os, pickle, numpy
"""
import os
import pickle
import numpy as np

# START: data/interim
# END:   data/processed
file_Path = os.path.abspath(__file__)
DATA_PATH = r"../../../data/interim\database.pkl"
OUTPUT_PATH = r"../../../data/processed/processed_data.pkl"
input_path = os.path.abspath(os.path.join(file_Path, DATA_PATH))
output_path = os.path.abspath(os.path.join(file_Path, OUTPUT_PATH))

if not os.path.exists(input_path):
        raise FileNotFoundError(f"The specified data folder does not exist: {input_path}")

# Open the database generated from the build_dataset script
with open(input_path, 'rb') as file:
    database = pickle.load(file)

# Only to verify if the database is working
names = [dict_db["name"] for dict_db in database if "name" in dict_db]
# print(names)

# You could choose the frequency in kHz or you could use
# the entire dataset
ALL_DATA = False
MY_FREQ = [40] #[kHz]
if ALL_DATA:
    my_database = database
else:
    my_database = []
    chosen_frequencies = [f'f{freq}' for freq in MY_FREQ]
    for dict in database:
        if any(dict["name"].endswith(suffix) for suffix in chosen_frequencies):
            my_database.append(dict)

data_dim = len(my_database)
shapes_db = []
for value in my_database[0].values():
    if isinstance(value, np.ndarray):
        shapes_db.append(value.shape)
    else:
        shapes_db.append(type(value))
print("\n")
print(f"The available Keys in the database are:\n"
    f"{my_database[0].keys()}\n"
    f"Shapes of values in the first dictionary: {shapes_db}")
print(f"\nThe dataset consist in {data_dim} curves\n")

# Extrapolating only the signal data that are passed as row vector
# so the dataset is a column vector
# As stated the time duration of the experiment is 1.31 *10^-3 [s]
# The Input of the model is a 12 transducer 12 sensors system,
# in fact the dimension of the input is dim_signal[1] * 66
signals_tot = []
temperature_number = []
DURATION = 0.00131 # total duration

# the dimension of the catch is the same for every frequency, picking the one
# of the first dictionary and creating the time vector then normalizing the input
dim_signal = np.shape(my_database[0]["catch"])
t = np.linspace(0, DURATION, dim_signal[1])
for dict in my_database:
    signals_tot.append(dict["catch"])
    # Creating a ordered list with the mean of the temperature (top and down plate)
    temperature_number.append(np.mean(dict["temperature"]))

length_catch = dim_signal[1]

print("Length of catch:", length_catch)

#Divide every matrix 66 x 13108 in a multiple vectors
signals = []
for matrix in signals_tot:
    single_signal = matrix[0, :]
    signals.append(single_signal)

output_data = {
    'signals': signals,
    'data_dim': data_dim,
    'length_catch': length_catch,
    'temperature_number': temperature_number,
    't': t
}

print("Output Path:", output_path)

if os.path.exists(output_path):
    print("The file already exist. Removing...")
    os.remove(output_path)

print("Creating the file processed_data.pkl")

# Saving the data into the pickle file
with open(output_path, 'wb') as output_file:
    pickle.dump(output_data, output_file)

print(f'Saved {output_path}')