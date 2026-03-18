import yfinance as yf
import pandas as pd
import numpy as np

# 1. Setup
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
start_date = '2023-01-01'

print("Downloading live data...")
# Download data - fetching only 'Close' as it is more reliable across tickers
raw_data = yf.download(tickers, start=start_date)

# Fix for the KeyError: Check if 'Adj Close' exists, otherwise use 'Close'
if 'Adj Close' in raw_data.columns.levels[0]:
    df = raw_data['Adj Close']
else:
    df = raw_data['Close']

# Cleanup: Drop failed tickers and handle missing values
df = df.dropna(axis=1, how='all')
df = df.ffill() # Fill any tiny gaps in data
active_tickers = df.columns.tolist()

# Re-align weights if any ticker failed
if len(active_tickers) < len(tickers):
    print(f"Adjusting weights for {len(active_tickers)} active tickers...")
    weights = np.array([1/len(active_tickers)] * len(active_tickers))

# 2. Returns and Volatility
returns = df.pct_change(fill_method=None).dropna()
volatility = returns.rolling(window=30).std() * np.sqrt(252)

# 3. Monte Carlo Simulation (10,000 runs)
def run_monte_carlo(returns, weights, days=252, iterations=10000):
    mean_ret = returns.mean()
    cov_matrix = returns.cov()
    
    port_mean = np.sum(mean_ret * weights)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Generate random daily returns
    sim_returns = np.random.normal(port_mean, port_std, (days, iterations))
    # Cumulative product to show price path starting at 1.0 (100%)
    portfolio_paths = np.cumprod(1 + sim_returns, axis=0)
    return portfolio_paths

print("Simulating 10,000 portfolio paths...")
sim_data = run_monte_carlo(returns, weights)

# 4. EXPORTING FOR POWER BI
print("Exporting CSVs...")
returns.corr().to_csv('correlation_matrix.csv')
volatility.to_csv('portfolio_vol.csv')

# Fan Chart Data
mc_projections = pd.DataFrame({
    'Day': range(252),
    'P5': np.percentile(sim_data, 5, axis=1),
    'Median': np.percentile(sim_data, 50, axis=1),
    'P95': np.percentile(sim_data, 95, axis=1)
})
mc_projections.to_csv('monte_carlo_projections.csv', index=False)

# Bell Curve Data (The last day of all 10,000 simulations)
bell_curve_df = pd.DataFrame(sim_data[-1, :], columns=['Final_Portfolio_Value'])
bell_curve_df.to_csv('bell_curve_data.csv', index=False)

print("Project AlphaPulse Data Ready!")