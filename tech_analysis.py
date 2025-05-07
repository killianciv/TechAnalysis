import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def get_bollinger_bands(ticker, period='6mo', interval='1d'):
    # Fetch historical data
    df = yf.download(ticker, period=period, interval=interval)

    # Calculate 20-day moving average and std dev
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()

    # Calculate upper and lower Bollinger Bands
    df['Upper'] = df['MA20'] + 2 * df['STD20']
    df['Lower'] = df['MA20'] - 2 * df['STD20']

    return df

def plot_bollinger_bands(df, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['MA20'], label='20-Day MA')
    plt.plot(df['Upper'], label='Upper Band', linestyle='--')
    plt.plot(df['Lower'], label='Lower Band', linestyle='--')
    plt.fill_between(df.index, df['Upper'], df['Lower'], color='gray', alpha=0.2)
    plt.title(f'Bollinger Bands for {ticker}')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Example usage
    ticker = 'AAPL'
    df = get_bollinger_bands(ticker)
    plot_bollinger_bands(df, ticker)

if __name__ == '__main__':
    main()


