"""
Inspect HDF5 File

This script provides a function to inspect the structure of an HDF5 file. 
It prints information about all items (groups, datasets, attributes) inside 
the file, including their names, types, shapes, and data types.

Author: Lorenzo Miele

Usage:
    1. Set the 'FILENAME' variable to the path of the HDF5 file you want to inspect.
    2. Run the script.

Note:
    - The 'print_hdf5_item' function is designed to be 
    used within a context manager (e.g., 'with h5py.File').
    
    - This script is intended for understanding the structure of 
    HDF5 files and extracting relevant features.
"""
import os
import h5py

def print_hdf5_item(name, item):
    """function that print out all the item 
    inside a hdf5 file type

    Args:
        name (string): filename
        item (hfd5 attribute): group, dataset, attribute etc.
    """
    print(f"{name}")
    if isinstance(item, h5py.Group):
        print("\tGroup")
    elif isinstance(item, h5py.Dataset):
        print("\tDataset")
    if isinstance(item, h5py.Dataset):
        print("\tShape:", item.shape)
        print("\tData type:", item.dtype)

script_dir = os.path.dirname(__file__)

FILEPATH = os.path.join(script_dir, '..', '..', 'data', 'raw', 'OGW_CFRP_Temperature_udam', '20181213T104208', 'pc_f40kHz.h5')

with h5py.File(FILEPATH, 'r') as file:
    file.visititems(print_hdf5_item)