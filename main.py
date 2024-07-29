import yfinance as yf
import matplotlib.pyplot as plt

# Function to fetch and plot stock data
def plot_stock_data(ticker, period, interval):
    # Fetch data from Yahoo Finance
    data = yf.download(ticker, period=period, interval=interval)
    print(data.head())
    # Plot the closing prices
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label=f'{ticker} Close Price')
    plt.title(f'{ticker} Stock Price ({period}, {interval})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
ticker = 'AAPL'  # Stock ticker symbol
period = '5y'    # Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', '10y', 'ytd', 'max')
interval = '1d'  # Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo')

plot_stock_data(ticker, period, interval)



def get_stock_data(ticker, period, interval):
    # Fetch data from Yahoo Finance
    data = yf.download(ticker, period=period, interval=interval)
    return data