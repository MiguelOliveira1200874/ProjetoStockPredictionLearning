import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Carregar o modelo treinado
model = load_model(r"StockPredictionModelo.keras")

st.header('Stock Price Prediction')

# Entrada do usuário para o ticker do estoque
user_input = st.text_input('Enter Stock Ticker', 'GOOG')
start = '2010-01-01'
today = datetime.today().strftime('%Y-%m-%d')

# Obter os dados do Yahoo Finance
data = yf.download(user_input, start, today)

# Mostrar os dados do estoque
st.subheader('Stock Data')
st.write(data)

# Plotar as médias móveis e os preços de fechamento
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

# Escalonar os dados para o intervalo [0, 1]
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['Close']])

# Preparar os dados de teste para previsão
sequence_length = 100
test_data = scaled_data[-sequence_length:]  # Usar os últimos 100 dias para gerar a previsão
batch = np.array([test_data])  # Converter em lote para o modelo

# Gerar previsões para cada dia útil até o final de 2024
end_date = '2024-12-31'
date_range = pd.date_range(start=data.index[-1], end=end_date, freq='B')
n_days = len(date_range) - 1  # Dias a partir de amanhã até o final de 2024
predictions = []

for i in range(n_days):
    current_pred = model.predict(batch)[0]
    predictions.append(current_pred)
    current_pred = current_pred.reshape(1, 1, 1)
    batch = np.append(batch[:, 1:], current_pred, axis=1)

# Reverter as previsões para os preços originais
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Criar um DataFrame para as previsões
predicted_index = date_range[1:]  # Começa no dia seguinte à última data conhecida
predictions_df = pd.DataFrame(predicted_prices, index=predicted_index, columns=['Prediction'])

# Plotando os resultados com as previsões
st.subheader('Extended Predictions till 2024')
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(data['Close'], label='Actual Close Price')
ax2.plot(predictions_df['Prediction'], label='Predicted Close Price', linestyle='--')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.set_title(f'Extended Stock Price Predictions for {user_input} till 2024')
ax2.legend()
st.pyplot(fig2)

# Avaliar o modelo com métricas de erro
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, mape

# Considerando que temos os dados reais (y_test) para comparar com as previsões (y_predicted)
# Como exemplo, vou comentar essas linhas, já que não temos y_test para o futuro
# mae, mse, mape = calculate_metrics(y_test, predicted_prices)
# st.subheader("Metrics Evaluation")
# st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
# st.write(f"Mean Squared Error (MSE): {mse:.2f}")
# st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
