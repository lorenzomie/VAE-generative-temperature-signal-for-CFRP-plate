# HOW TO CLEAN INSTALL

After the repository is cloned locally, follow these steps to build the code. Make sure to have 80 GB of free memory since the dataset is large.

1. **Create your virtual environment with CONDA or with venv as you prefer:**

    - **CONDA:**
        ```bash
        conda create --name venv_name python=3.8
        conda activate venv_name
        ```

    - **WITHOUT CONDA:**
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
   WARNING: Since the data to process is very large, you may have to run manually `src/data/make_dataset` and then `src
