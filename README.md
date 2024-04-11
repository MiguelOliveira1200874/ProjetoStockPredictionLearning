# Stock Price Prediction App

This application predicts the future stock prices of a given company using a deep learning model trained on historical stock data. In this example, the model has been trained on Google (Alphabet Inc.) stock data.

## Requirements

- Python 3.x
- Libraries listed in `requirements.txt`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/MiguelOliveira1200874/ProjetoStockPredictionLearning.git
    cd stock-price-prediction
    ```

2. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Enter the stock ticker symbol of the company you want to predict.

3. Explore the stock data, including moving averages and predicted vs actual close prices.

4. View metrics evaluation including Mean Absolute Error (MAE), Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE).

## Model Training

- The deep learning model used for stock price prediction was trained on historical Google (Alphabet Inc.) stock data.

- The model was trained using the Keras library with TensorFlow backend.

- Training data included historical stock prices, which were preprocessed and scaled before feeding into the model.

- The model architecture, training process, and hyperparameters may vary depending on the specific implementation.

## Disclaimer

This application is for educational and demonstration purposes only. Stock price prediction is inherently uncertain and should not be used for making financial decisions without consulting professional advice.

