HOW TO CLEAN INSTALL
======================

After the repository is cloned on local there are a few steps to follow in order
to build the code. Make sure to have 80 GB of free memory since the dataset is large.

1. create your virtual environment with CONDA or with venv as you prefer
    In your terminal:
    CONDA: create your virtual environment, substitute venv_name with your preferred name
    conda create --name venv_name python=3.8
    Activate the virtual environment
    conda activate venv_name

    WITHOUT CONDA: Once you are in your project folder, create the virtual environment substituting venv_name with your desired name
    python3 -m venv venv_name

    Activate the virtual environment
    source venv_name/bin/activate

3. Check your virtual environment with:
    make test_environment

4. Install the dependencies with:
    make requirements
    It require a few minutes

5. Download https://springernature.figshare.com/articles/dataset/OGW_CFRP_Temperature_udam/8108297?backTo=/collections/Temperature_affected_guided_wave_propagation_in_a_composite_plate_complementing_the_Open_Guided_Waves_Platform/4488089
    in your data/raw folder. It have to figure data/raw/OGW_CFRP_Temperature_udam.
    It will require some time

6. To process data type:
    make data
    WARNING: since the data to process are very large. maybe you have to run manually src/data/make dataset and then src/features/build_features script in this order

7. To Build and Train a model:
    make create_model
    
    Here you could choose between 3 models of a Variational Autoencoder
    STANDARD: it will use all the dataset data in terms of temperature (20 to 60 degrees)
    BAND: it will use only a band of temperature. You have to modify the temperatures with MIN_TEMP and   MAX_TEMP  (30/50 by default)
    SPARSE: it will use clusters of signal with similar temperature. You could change the distance of the clusters and the radius of the temperature neighbourhood
    More info available in the code

8. To test one model created:
    make test_model

    Here you could modify the TEMPERATURE constant in the script to generate a signal at a different temperature

You should be ready to create your own Variational Autoencoder.

The Makefile could execute the following command on your linux terminal with make:

- clean:               Delete all compiled Python files
- data:                Make Dataset
- create_model:        Build and train a model
- test_model:          Test a model once created
- lint:                Lint using flake8
- requirements:        Install Python Dependencies
- test_environment:    Test python environment is setup correctly
