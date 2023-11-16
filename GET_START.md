# HOW TO CLEAN INSTALL

After the repository is cloned locally, follow these steps to build the code. Make sure to have 80 GB of free memory since the dataset is large.

1. **Create your virtual environment with CONDA or with venv as you prefer:**

    - **CONDA: change venv_name in yout desired environment name**
        ```bash
        conda create --name venv_name python=3.8
        conda activate venv_name
        ```

    - **WITHOUT CONDA: change venv_name in yout desired environment name**
        ```bash
        python3 -m venv venv_name
        source venv_name/bin/activate
        ```

2. **Check your virtual environment with:**
    ```bash
    make test_environment
    ```

3. **Install the dependencies with:**
    ```bash
    make requirements
    ```
   It requires a few minutes.

4. **Download the dataset:**
   [OGW_CFRP_Temperature_udam Dataset](https://springernature.figshare.com/articles/dataset/OGW_CFRP_Temperature_udam/8108297?backTo=/collections/Temperature_affected_guided_wave_propagation_in_a_composite_plate_complementing_the_Open_Guided_Waves_Platform/4488089)
   
   Place the downloaded file in your `data/raw` folder, and it should appear as `data/raw/OGW_CFRP_Temperature_udam`. This step may take some time.

5. **To process data type:**
    ```bash
    make data
    ```
   WARNING: Since the data to process is very large, you may have to run manually `src/data/make_dataset` and then `src/features/build_features` script in this order.

6. **To Build and Train a model:**
    ```bash
    make create_model
    ```

    Here you could choose between 3 models of a Variational Autoencoder:
    - STANDARD: It will use all the dataset data in terms of temperature (20 to 60 degrees).
    - BAND: It will use only a band of temperature. You have to modify the temperatures with MIN_TEMP and MAX_TEMP (30/50 by default).
    - SPARSE: It will use clusters of signal with similar temperature. You could change the distance of the clusters and the radius of the temperature neighborhood.
    
    More info available in the code.

7. **To test one model created:**
    ```bash
    make test_model
    ```

    Here you could modify the TEMPERATURE constant in the script to generate a signal at a different temperature.

You should be ready to create your own Variational Autoencoder.

The Makefile could execute the following command on your linux terminal with make:

- `clean`: Delete all compiled Python files
- `data`: Make Dataset
- `create_model`: Build and train a model
- `test_model`: Test a model once created
- `lint`: Lint using flake8
- `requirements`: Install Python Dependencies
- `test_environment`: Test python environment is set up correctly
