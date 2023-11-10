from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Utilizing OpenGuided Waves dataset, this project involves pitch-catch values corresponding to Lamb waves on a carbon fiber plate at various temperatures. Implementing a Variational Autoencoder (VAE), the aim is to generate missing signals in the dataset based on user input for the desired temperature."',
    author='Lorenzo Miele',
    license='MIT',
)
