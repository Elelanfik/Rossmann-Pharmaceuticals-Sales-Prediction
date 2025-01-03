# Rossmann-Pharmaceuticals-Sales-Prediction

## Overview
This repository contains the code and documentation for the Rossmann Store Sales Prediction project, which focuses on exploring customer purchasing behavior and predicting future sales using machine learning and deep learning techniques. The project is structured around three main tasks:

1. **Exploratory Data Analysis (EDA)**
2. **Sales Prediction using LSTM**
3. **Serving Predictions via a Web Interface**

## Table of Contents
- [Project Description](#project-description)
- [Tasks](#tasks)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description
The goal of this project is to analyze the Rossmann Store Sales dataset to understand customer purchasing behavior and to build predictive models that can forecast future sales. The project utilizes various data science techniques, including data cleaning, exploratory analysis, machine learning, and deep learning.

## Tasks

### Task 1: Exploratory Data Analysis (EDA)
- Conducted a thorough analysis of customer purchasing behavior.
- Cleaned the dataset by handling outliers and missing values.
- Visualized key features and their interactions with sales.
- Explored the impact of promotions and store operations on sales.

### Task 2: Sales Prediction using LSTM
- Isolated the dataset into time series data.
- Prepared the data for modeling by checking for stationarity and transforming it into a supervised learning format.
- Built and trained a Long Short-Term Memory (LSTM) regression model using TensorFlow/PyTorch.
- Evaluated model performance and serialized the model for future predictions.

### Task 3: Serving Predictions via a Web Interface
- Developed a REST API to serve the trained LSTM model for real-time predictions.
- Implemented endpoints to accept input data and return predictions.
- Deployed the API to a web server/cloud platform for accessibility.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn (for visualization)
- TensorFlow / PyTorch (for deep learning)
- Flask / FastAPI (for API development)
- Jupyter Notebook (for analysis and documentation)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Elelanfik/Rossmann-Pharmaceuticals-Sales-Prediction
   cd Rossmann-Pharmaceuticals-Sales-Prediction

1. Install the required packages:
pip install -r requirements.txt

Usage

- For EDA, run the Jupyter Notebook located in the eda directory.

- To train the LSTM model, navigate to the lstm_model directory and run the corresponding script.

- To serve the model, go to the api directory and run the API server.

Contributing

Contributions are welcome! If you have suggestions for improvements or want to add features, please fork the repository and submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

- Special thanks to the 10 Academy for providing the framework and resources for this project.

- Dataset sourced from Kaggle.