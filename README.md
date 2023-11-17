Signal Generation VAE
==============================

Utilizing OpenGuided Waves dataset, this project involves pitch-catch values corresponding to Lamb waves on a carbon fiber plate at various temperatures. Implementing a Variational Autoencoder (VAE), the aim is to generate missing signals in the dataset based on user input for the desired temperature."

What you can do?
------------

This model is a Variational Autoencoder with a architecture made by dense layers. It will utilize a free dataset uploaded in openguidedwaves (more info on GET_START) to build a model capable of generate lamb waves signal at a desired TEMPERATURE

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             
    │   ├── model_data     <- Models data required for visualized
    │   └── weight         <- Trained and serialized model weight
    │       ├── band       <- model trained with all the temperature in the dataset
    │       ├── sparse     <- model trained with clusters of temperature in the dataset
    │       └── standard   <- model trained with all the temperature in the dataset
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │   └── print_h5_tree.py <- Generate the h5 tree to understand the structure of the dataset
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to turn raw data into python list
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn python list into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── predict_model.py
    │       └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
