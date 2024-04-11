import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = load_model(r"StockPredictionModelo.keras")

st.header('Stock Price Prediction')

# Entrada do usuário para o ticker do estoque
user_input = st.text_input('Enter Stock Ticker', 'GOOG')
start = '2010-01-01'
end = '2024-11-04'

# Obter os dados do Yahoo Finance
data = yf.download(user_input, start, end)

# Mostrar os dados do estoque
st.subheader('Stock Data')
st.write(data)

# Dividir os dados em conjunto de treinamento e teste
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80): int(len(data))])

# Escalonar os dados para o intervalo [0, 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# Adicionar os últimos 100 dias do conjunto de treinamento ao conjunto de teste
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

# Mostrar a média móvel de 50 dias (MA50)
st.subheader("MA50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'b', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('MA50 vs Close Price')
st.pyplot(fig1)

# Mostrar a média móvel de 100 dias (MA100)
st.subheader("MA100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days, 'r', label='MA100')
plt.plot(data.Close, 'b', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('MA100 vs Close Price')
st.pyplot(fig2)

# Mostrar a média móvel de 200 dias (MA200)
st.subheader("MA200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(12,6))
plt.plot(ma_200_days, 'r', label='MA200')
plt.plot(data.Close, 'b', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('MA200 vs Close Price')
st.pyplot(fig3)




# Mostrar a média móvel de 50 dias (MA50) vs 100 dias (MA100) vs 200 dias (MA200) vs Close Price
st.subheader("MA50 vs MA100 vs MA200 vs Close Price")
fig4 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'g', label='MA100')
plt.plot(ma_200_days, 'y', label='MA200')
plt.plot(data.Close, 'b', label='Close Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('MA50 vs MA100 vs MA200 vs Close Price')
st.pyplot(fig4)

# Preparar os dados de teste para previsão
x_test = []
y_test = []

# Criar janelas de 100 dias para a previsão
for i in range(100, data_test_scaler.shape[0]):
    x_test.append(data_test_scaler[i-100: i])
    y_test.append(data_test_scaler[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Fazer a previsão usando o modelo carregado
y_predicted = model.predict(x_test)

# Escalar os dados de volta ao intervalo original
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Mostrar os dados previstos e reais
st.subheader("Predicted vs Actual Close Price")
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Actual Close Price')
plt.plot(y_predicted, 'r', label='Predicted Close Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Predicted vs Actual Close Price')
st.pyplot(fig3)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# Função para calcular o Mean Absolute Error (MAE)
def calculate_mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae

# Função para calcular o Mean Squared Error (MSE)
def calculate_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

# Função para calcular o Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mape

# Calcular as métricas de avaliação
mae = calculate_mae(y_test, y_predicted)
mse = calculate_mse(y_test, y_predicted)
mape = calculate_mape(y_test, y_predicted)

# Exibir as métricas e suas explicações
st.subheader("Metrics Evaluation")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")