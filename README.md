# Multiple Disease Prediction System

## Overview
The Multiple Disease Prediction System is a web application built with Python and Streamlit. It enables users to predict the likelihood of four diseases: Diabetes, Heart Disease, Breast Cancer, and Parkinson's Disease. By inputting relevant data, users receive instant predictions, aiding in early disease detection and management.

## Features
- Predicts multiple diseases: Diabetes, Heart Disease, Breast Cancer, and Parkinson's Disease.
- User-friendly interface for inputting relevant medical data.
- Utilizes trained machine learning models for accurate predictions.
- Displays clear results indicating the presence or absence of each disease.
- Allows users to interactively explore different prediction scenarios.

## Technologies Used
This project utilizes the following technologies, frameworks, and libraries:

- **Python**: The primary programming language used for developing the machine learning models and backend logic.
- **Streamlit**: A Python library used for creating interactive web applications with simple Python scripts.
- **Pandas**: A powerful data manipulation library used for data preprocessing and analysis.
- **Scikit-learn**: A machine learning library in Python used for building and training predictive models.
- **Pickle**: Python module used for serializing and deserializing Python objects, particularly machine learning models.
- **GitHub**: Used for version control and collaboration among team members.
- **Jupyter Notebook**: Utilized for exploratory data analysis, model development, and documentation.
- **HTML/CSS**: Used for customizing the appearance and layout of the Streamlit web application.
- **Markdown**: Utilized for writing formatted text, such as this README file.
- **NumPy**: A Python library used for numerical computing, particularly for handling arrays and matrices.

## Installation
To run the application locally, follow these steps:

1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/deepgoenka/Multiple-Disease-Prediction-System.git
   ```

2. Install the required dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application.
   ```bash
   streamlit run MultipleDiseasePrediction.py
   ```

4. Access the application in your web browser at `http://localhost:8501`.

## Usage
- Upon running the application, you will be presented with a sidebar menu to select the disease prediction task.
- Choose the disease you want to predict from the options provided: Diabetes Prediction, Heart Disease Prediction, Breast Cancer Classification, or Parkinson's Disease Prediction.
- Input the required medical data into the form fields provided.
- Click the "Predict" button to see the prediction results.
- The application will display a message indicating whether the predicted disease is present or absent based on the provided input data.

## Results
The performance of the machine learning models on the test dataset is as follows:

* **Diabetes Prediction:**
  - Model: Random Forest Classifier
  - Accuracy: 99.04%

* **Heart Disease Prediction:**
  - Model: Random Forest Classifier
  - Accuracy: 98.54%

* **Breast Cancer Classification:**
  - Model: Logistic Regression
  - Accuracy: 98.25%

* **Parkinson's Disease Prediction:**
  - Model: Support Vector Machine
  - Accuracy: 90%

These results demonstrate the effectiveness of the trained models in accurately predicting the presence or absence of various diseases based on the input features provided.

## Contributing
Contributions are welcome! Please feel free to submit bug reports, feature requests, or pull requests to help improve this project.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/deepgoenka/Multiple-Disease-Prediction-System/blob/main/LICENSE) file for details.
