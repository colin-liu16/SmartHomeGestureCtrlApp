# SmartHome Gesture Control ML


## Project Setup

This project requires Python 3.8 and specific Python modules to be installed.
Required Modules:

    TensorFlow (v2.12.0)
    Keras (v2.12.0)
    OpenCV for Python (v4.8.0.74)
    Numpy (v1.23.4)
    Pandas (v1.5.3)
    Scikit-learn (v1.2.1)

You can install these dependencies by running:

bash

pip install -r requirements.txt

## IDE:

You can use any Python-based IDE for this project.
Python Version:

Ensure you are using Python 3.8, as this is the supported version for the project.
Run Locally

### To run the project locally:

    Place the test videos in the test folder.
    Place the training videos in the traindata folder.
    Run the program with the following command:

    bash

    python3 main.py

    The program will extract frames and save them in a new frames folder within each respective video folder.
    The gesture recognition results will be saved in results.csv. The output file will have a header with a column labeled Output Label.

This text provides clear instructions and includes how to install the required modules from the requirements.txt file. You can now copy and paste it as needed.

