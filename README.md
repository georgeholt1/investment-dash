# Investment Return Dashboard

This project contains a [Dash](https://plotly.com/dash/) application that can be used to calculate and visualise the future value of an investment, including additional contributions and interest, over a given period of time. 

The investment settings are controlled through the dashboard interface, which also dynamically shows the resulting balance and its breakdown over time in interactive plots.

## Installation

### Installing for development

Development dependencies are handled using [conda](https://docs.conda.io/en/latest/) and may be installed by running `conda env create -f conda_env.yml`.

Pre-commit hooks for formatting code with black and isort can be installed by running `pre-commit install` from the repository root directory.

To run the application in debug mode: `python app.py --debug`