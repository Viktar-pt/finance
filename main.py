# !!!!!  pyenv activate trading-env
# pip install numpy pandas tensorflow scikit-learn matplotlib requests


import numpy as np
import pandas as pd
import requests
import time
import os
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime


SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LOOK_BACK = 60
TRAIN_TEST_SPLIT = 0.8
MODEL_FILE = "btc_lstm_model.h5"
SCALER_FILE = "btc_scaler.save"

# convert interval to milliseconds
interval_ms = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000
}


def get_all_klines(symbol, interval="1h", limit=1000, max_candles=5000):
    """included pagination"""
    print("Загрузка данных с Binance...")
    _url = "https://api.binance.com/api/v3/klines"
    df = pd.DataFrame()
    start_time = int(time.time() * 1000) - interval_ms.get(interval, ) * max_candles

    while len(df) < max_candles:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_time}
        response = requests.get(_url, params=params)
        data = response.json()

        if not data:
            break

        batch = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        batch["timestamp"] = pd.to_datetime(batch["timestamp"], unit="ms")
        batch[["open", "high", "low", "close"]] = batch[["open", "high", "low", "close"]].astype(float)

        df = pd.concat([df, batch[["timestamp", "close"]]], ignore_index=True)
        start_time = int(batch["timestamp"].iloc[-1].timestamp() * 1000) + interval_ms[interval]
        #  as I use public indpoints I have restricted access and should
        #  take care about timeouts
        time.sleep(0.5)

        if len(batch) < limit:
            break

    print(f"Загружено {len(df)} свечей.")
    return df


# preparing DATA for LTSM
def prepare_data(data, look_back, train_split):
    close_prices = data[["close"]].values

    split_index = int(train_split * len(close_prices))
    train_data = close_prices[:split_index]
    test_data = close_prices[split_index:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    scaled_all = scaler.transform(close_prices)

    X, y = [], []
    for i in range(look_back, len(scaled_all)):
        X.append(scaled_all[i - look_back:i, 0])
        y.append(scaled_all[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(train_split * len(X))
    return X[:split], y[:split], X[split:], y[split:], scaler



def create_model(look_back):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def plot_results(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Реальная цена", color="blue")
    plt.plot(predicted, label="Прогноз", color="red", alpha=0.7)
    plt.title("Прогнозирование цены BTC с помощью LSTM")
    plt.xlabel("Время")
    plt.ylabel("Цена (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_model(model, scaler):
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Модель сохранена в {MODEL_FILE}, scaler — в {SCALER_FILE}")


def load_saved_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    return None, None


def main(train_more=False):
    data = get_all_klines(SYMBOL, INTERVAL, max_candles=5000)

    X_train, y_train, X_test, y_test, scaler = prepare_data(data, LOOK_BACK, TRAIN_TEST_SPLIT)

    if train_more:
        print("Загрузка сохранённой модели...")
        model, saved_scaler = load_saved_model()
        if model is None:
            print("Сохранённая модель не найдена.")
            return
    else:
        print("Создание и обучение новой модели...")
        model = create_model(LOOK_BACK)

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    predictions = model.predict(X_test)
    predictions_actual = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    plot_results(y_test_actual, predictions_actual)

    save_model(model, scaler)

    # Forecast the next price
    last_sequence = data["close"].values[-LOOK_BACK:]
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    last_sequence_scaled = np.reshape(last_sequence_scaled, (1, LOOK_BACK, 1))

    predicted_price_scaled = model.predict(last_sequence_scaled)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

    print(f"\nПоследняя известная цена: {data["close"].iloc[-1]:.2f} USD")
    print(f"Прогнозируемая цена: {predicted_price:.2f} USD")

if __name__ == "__main__":
    # Запуск с дообучением: main(train_more=True)
    main(train_more=False)  #  запуск без дообучения


