import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Carregar o modelo treinado
model = load_model("StockPredictionModelo.keras")

st.header('Previsão do Preço de Ações')

# Entrada do usuário para o ticker do estoque
user_input = st.text_input('Digite o Ticker do Estoque', 'GOOG')
start = '2010-01-01'
end = datetime.today().strftime('%Y-%m-%d')

# Obter os dados do Yahoo Finance
data = yf.download(user_input, start, end)
data_close = data['Close'].values.reshape(-1,1)

# Mostrar os dados do estoque
st.subheader('Dados Históricos de Fechamento')
st.line_chart(data['Close'])

# Preparar os dados para previsão usando MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_close)

# Definir o tamanho da janela para criar sequência
sequence_length = 100

# Dados de teste para fazer previsões futuras
last_sequence = scaled_data[-sequence_length:]
batch = np.array([last_sequence])  # Ensure batch is the correct shape

# Calcular quantos dias úteis até o final de 2024
last_known_date = data.index[-1]
end_date = '2024-12-31'
date_range = pd.date_range(start=last_known_date, end=end_date, freq='B')
n_days = len(date_range) - 1  # Days from tomorrow until end of 2024

# Gerar previsões para cada dia útil até o final de 2024
predictions = []
for i in range(n_days):
    current_pred = model.predict(batch)[0]
    predictions.append(current_pred)
    current_pred = current_pred.reshape(1, 1, 1)  # Correct reshaping
    batch = np.append(batch[:, 1:], current_pred, axis=1)  # Proper appending

# Reverter as previsões para os preços originais
predictions = np.array(predictions).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predictions)

# Certificar de que as previsões e o index estão alinhados
predicted_prices = predicted_prices[:len(date_range)]  # Ajustar para ter o mesmo número de previsões que o intervalo de datas

# Criar um DataFrame para as previsões
predictions_df = pd.DataFrame(predicted_prices, index=date_range[:-1], columns=['Prediction'])  # Excluindo a data de início de 2025

# Plotando os resultados
st.subheader('Previsões Extendidas Até o Final de 2024')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Close'], label='Preço de Fechamento Atual')
ax.plot(predictions_df['Prediction'], label='Preço de Fechamento Previsto', linestyle='--')
ax.set_xlabel('Data')
ax.set_ylabel('Preço')
ax.set_title(f'Previsões de Preço de Ações para {user_input} Até o Final de 2024')
ax.legend()
st.pyplot(fig)
