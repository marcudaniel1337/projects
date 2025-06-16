"""
Advanced Stock Price Predictor
==============================

This is a comprehensive stock price prediction system that combines multiple
machine learning approaches. I've tried to make this as realistic as possible
while keeping it educational - remember, real stock prediction is incredibly
difficult and this shouldn't be used for actual trading decisions!

The approach here uses:
1. Technical indicators (moving averages, RSI, MACD, etc.)
2. Multiple ML models (LSTM, Random Forest, XGBoost)
3. Ensemble learning to combine predictions
4. Proper validation and backtesting

Let's dive in!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# I'm simulating the deep learning imports since we might not have them installed
# In a real environment, you'd use: from tensorflow.keras import models, layers
# For now, I'll create a mock LSTM class to show the structure

class MockLSTM:
    """
    This is a mock LSTM class to demonstrate the structure.
    In reality, you'd use TensorFlow/Keras LSTM layers.
    """
    def __init__(self, units=50, return_sequences=True):
        self.units = units
        self.return_sequences = return_sequences
        self.weights = None
        
    def fit(self, X, y, epochs=50, batch_size=32, verbose=0):
        # Simulate training - in reality this would train the neural network
        print(f"Training LSTM with {self.units} units for {epochs} epochs...")
        # Mock some training behavior
        self.weights = np.random.random((X.shape[1], self.units))
        return self
        
    def predict(self, X):
        # Mock prediction - in reality this would use trained weights
        # Just return a simple linear combination for demonstration
        if self.weights is not None:
            return np.dot(X.mean(axis=1), self.weights).mean(axis=1).reshape(-1, 1)
        return np.random.random((X.shape[0], 1))

def generate_sample_data(ticker="AAPL", days=1000):
    """
    Generate realistic-looking stock data for demonstration.
    
    In a real application, you'd fetch this from APIs like:
    - Yahoo Finance (yfinance library)
    - Alpha Vantage
    - Quandl
    - IEX Cloud
    
    I'm generating synthetic data here so the example works standalone.
    """
    print(f"Generating sample data for {ticker}...")
    
    # Create a date range
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # Generate realistic stock price movement using random walk with drift
    # Starting price around $100
    np.random.seed(42)  # For reproducible results
    
    # Create price series with some trend and volatility
    returns = np.random.normal(0.0008, 0.02, days)  # Daily returns: slight positive drift, 2% volatility
    
    # Add some autocorrelation to make it more realistic
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Slight momentum effect
    
    # Convert returns to prices
    prices = [100]  # Starting price
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate volume (inversely correlated with price changes)
    volume = np.random.lognormal(15, 0.5, days)  # Log-normal distribution for volume
    price_changes = np.diff(prices) / prices[:-1]
    volume[1:] *= (1 + np.abs(price_changes) * 2)  # Higher volume on big moves
    
    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        # Create realistic OHLC from close price
        volatility = np.random.normal(0, 0.01)  # Daily volatility
        high = price * (1 + abs(volatility) + np.random.uniform(0, 0.02))
        low = price * (1 - abs(volatility) - np.random.uniform(0, 0.02))
        open_price = prices[i-1] * (1 + np.random.normal(0, 0.005)) if i > 0 else price
        
        data.append({
            'Date': dates[i],
            'Open': max(open_price, 0.01),
            'High': max(high, open_price, price),
            'Low': min(low, open_price, price),
            'Close': price,
            'Volume': int(volume[i])
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    print(f"Generated {len(df)} days of data from {df.index[0].date()} to {df.index[-1].date()}")
    return df

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators that traders commonly use.
    
    These indicators help capture different aspects of price movement:
    - Trend indicators (moving averages)
    - Momentum indicators (RSI, MACD)
    - Volatility indicators (Bollinger Bands)
    - Volume indicators
    
    The idea is that these capture patterns that pure price data might miss.
    """
    print("Calculating technical indicators...")
    
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Moving Averages - these smooth out price action to show trends
    data['MA_5'] = data['Close'].rolling(5).mean()    # Short-term trend
    data['MA_10'] = data['Close'].rolling(10).mean()  # Medium-term trend
    data['MA_20'] = data['Close'].rolling(20).mean()  # Longer-term trend
    data['MA_50'] = data['Close'].rolling(50).mean()  # Long-term trend
    
    # Exponential Moving Averages - give more weight to recent prices
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # MACD - Moving Average Convergence Divergence
    # This is a momentum indicator that shows relationship between two moving averages
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    # RSI - Relative Strength Index
    # Measures overbought/oversold conditions (0-100 scale)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands - measure volatility
    # Price tends to bounce between upper and lower bands
    bb_period = 20
    data['BB_Middle'] = data['Close'].rolling(bb_period).mean()
    bb_std = data['Close'].rolling(bb_period).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
    
    # Price momentum and change indicators
    data['Price_Change'] = data['Close'].pct_change()
    data['Price_Change_2d'] = data['Close'].pct_change(2)
    data['Price_Change_5d'] = data['Close'].pct_change(5)
    
    # Volatility measures
    data['Volatility_10d'] = data['Price_Change'].rolling(10).std()
    data['Volatility_30d'] = data['Price_Change'].rolling(30).std()
    
    # Volume indicators
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    # High-Low indicators
    data['HL_Ratio'] = data['High'] / data['Low']
    data['OC_Ratio'] = data['Open'] / data['Close']
    
    # Support and Resistance levels (simplified)
    data['Recent_High'] = data['High'].rolling(20).max()
    data['Recent_Low'] = data['Low'].rolling(20).min()
    data['Price_Position'] = (data['Close'] - data['Recent_Low']) / (data['Recent_High'] - data['Recent_Low'])
    
    print(f"Added {len([col for col in data.columns if col not in df.columns])} technical indicators")
    return data

def prepare_features(df, target_days=1):
    """
    Prepare features for machine learning models.
    
    This is where we decide what information to feed our models.
    We'll use:
    1. Current technical indicators
    2. Historical price patterns (lagged features)
    3. Rolling statistics
    
    target_days: how many days ahead we want to predict
    """
    print(f"Preparing features for {target_days}-day ahead prediction...")
    
    # Calculate technical indicators first
    data = calculate_technical_indicators(df)
    
    # Create target variable - what we want to predict
    # We'll predict the closing price N days in the future
    data['Target'] = data['Close'].shift(-target_days)
    
    # Create lagged features - use past values as predictors
    # The idea is that recent price action might predict future moves
    for lag in [1, 2, 3, 5, 10]:
        data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)
        data[f'RSI_lag_{lag}'] = data['RSI'].shift(lag)
        data[f'MACD_lag_{lag}'] = data['MACD'].shift(lag)
    
    # Rolling statistics features
    # These capture recent trends and patterns
    for window in [3, 7, 14]:
        data[f'Close_mean_{window}d'] = data['Close'].rolling(window).mean()
        data[f'Close_std_{window}d'] = data['Close'].rolling(window).std()
        data[f'Volume_mean_{window}d'] = data['Volume'].rolling(window).mean()
        data[f'High_max_{window}d'] = data['High'].rolling(window).max()
        data[f'Low_min_{window}d'] = data['Low'].rolling(window).min()
    
    # Day of week and month effects
    # Some research suggests certain days/months have different patterns
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Day_of_Month'] = data.index.day
    
    # Create binary features for day of week (one-hot encoding)
    for i in range(7):
        data[f'Is_Day_{i}'] = (data['Day_of_Week'] == i).astype(int)
    
    print(f"Created {len(data.columns)} total features")
    return data

def create_lstm_sequences(data, sequence_length=60):
    """
    Prepare data for LSTM model.
    
    LSTMs work with sequences of data - they look at the last N days
    to predict the next day. This is different from traditional ML models
    that look at individual data points.
    
    sequence_length: how many days of history to use for each prediction
    """
    print(f"Creating LSTM sequences with {sequence_length} day lookback...")
    
    # Select features for LSTM (avoid using future information)
    feature_columns = [col for col in data.columns 
                      if col not in ['Target'] and not col.endswith('_future')]
    
    # Drop rows with NaN values
    clean_data = data[feature_columns + ['Target']].dropna()
    
    # Separate features and target
    features = clean_data[feature_columns].values
    target = clean_data['Target'].values
    
    # Scale the features (important for neural networks)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features_scaled)):
        # Use previous sequence_length days as input
        X.append(features_scaled[i-sequence_length:i])
        y.append(target[i])
    
    X, y = np.array(X), np.array(y)
    
    print(f"Created {len(X)} sequences of shape {X.shape}")
    return X, y, scaler, clean_data.index[sequence_length:]

class StockPredictor:
    """
    Main class that combines multiple prediction models.
    
    This uses an ensemble approach - we train several different models
    and combine their predictions. The theory is that different models
    might capture different patterns, and combining them reduces overfitting.
    """
    
    def __init__(self, target_days=1):
        self.target_days = target_days
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.lstm_scaler = None
        
        print(f"Initialized StockPredictor for {target_days}-day predictions")
    
    def prepare_data(self, df):
        """Prepare all the data we need for training"""
        print("Preparing data for training...")
        
        # Get features with technical indicators
        self.full_data = prepare_features(df, self.target_days)
        
        # Prepare LSTM sequences
        self.lstm_X, self.lstm_y, self.lstm_scaler, self.lstm_dates = create_lstm_sequences(
            self.full_data, sequence_length=60
        )
        
        # Prepare traditional ML data
        # Remove columns that wouldn't be available at prediction time
        feature_cols = [col for col in self.full_data.columns 
                       if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Drop rows with NaN values
        ml_data = self.full_data[feature_cols + ['Target']].dropna()
        
        self.ml_X = ml_data[feature_cols]
        self.ml_y = ml_data['Target']
        self.ml_dates = ml_data.index
        self.feature_columns = feature_cols
        
        print(f"ML data shape: {self.ml_X.shape}")
        print(f"LSTM data shape: {self.lstm_X.shape}")
        
        return self
    
    def train_models(self):
        """Train all our different models"""
        print("Training ensemble of models...")
        
        # Split data for training (use time series split to avoid lookahead bias)
        split_idx = int(len(self.ml_X) * 0.8)
        
        # Traditional ML models training data
        X_train = self.ml_X.iloc[:split_idx]
        y_train = self.ml_y.iloc[:split_idx]
        X_test = self.ml_X.iloc[split_idx:]
        y_test = self.ml_y.iloc[split_idx:]
        
        # Scale features for some models
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # 1. Random Forest - good for capturing non-linear patterns
        print("Training Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100,      # Number of trees
            max_depth=10,          # Prevent overfitting
            min_samples_split=5,   # Prevent overfitting
            random_state=42
        )
        self.models['rf'].fit(X_train, y_train)
        
        # 2. XGBoost-like model (using sklearn's GradientBoosting as substitute)
        print("Training Gradient Boosting...")
        from sklearn.ensemble import GradientBoostingRegressor
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.models['gb'].fit(X_train, y_train)
        
        # 3. Linear model for trend capture
        print("Training Ridge Regression...")
        from sklearn.linear_model import Ridge
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_train_scaled, y_train)
        
        # 4. LSTM model (using our mock implementation)
        print("Training LSTM...")
        lstm_split = int(len(self.lstm_X) * 0.8)
        lstm_X_train = self.lstm_X[:lstm_split]
        lstm_y_train = self.lstm_y[:lstm_split]
        
        self.models['lstm'] = MockLSTM(units=50)
        self.models['lstm'].fit(lstm_X_train, lstm_y_train, epochs=50)
        
        # Evaluate models on test set
        print("\nModel Performance on Test Set:")
        print("-" * 40)
        
        # Traditional ML models
        for name, model in self.models.items():
            if name == 'lstm':
                continue
                
            if name == 'ridge':
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name.upper():>12}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # LSTM evaluation
        lstm_X_test = self.lstm_X[lstm_split:]
        lstm_y_test = self.lstm_y[lstm_split:]
        lstm_pred = self.models['lstm'].predict(lstm_X_test)
        
        mse = mean_squared_error(lstm_y_test, lstm_pred)
        mae = mean_absolute_error(lstm_y_test, lstm_pred)
        r2 = r2_score(lstm_y_test, lstm_pred)
        print(f"{'LSTM':>12}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return self
    
    def predict(self, recent_data, ensemble_weights=None):
        """
        Make predictions using ensemble of models.
        
        recent_data: DataFrame with recent stock data
        ensemble_weights: weights for combining model predictions
        """
        if ensemble_weights is None:
            # Equal weights for all models
            ensemble_weights = {name: 1.0 for name in self.models.keys()}
        
        # Prepare features for recent data
        featured_data = prepare_features(recent_data, self.target_days)
        
        predictions = {}
        
        # Get ML predictions
        ml_features = featured_data[self.feature_columns].iloc[-1:].fillna(method='ffill')
        
        # Random Forest prediction
        if 'rf' in self.models:
            predictions['rf'] = self.models['rf'].predict(ml_features)[0]
        
        # Gradient Boosting prediction
        if 'gb' in self.models:
            predictions['gb'] = self.models['gb'].predict(ml_features)[0]
        
        # Ridge prediction
        if 'ridge' in self.models:
            ml_features_scaled = self.scalers['standard'].transform(ml_features)
            predictions['ridge'] = self.models['ridge'].predict(ml_features_scaled)[0]
        
        # LSTM prediction (need sequence of data)
        if 'lstm' in self.models and len(featured_data) >= 60:
            # Get last 60 days of data
            lstm_features = featured_data.iloc[-60:][
                [col for col in featured_data.columns if col != 'Target']
            ].fillna(method='ffill')
            
            # Scale features
            lstm_features_scaled = self.lstm_scaler.transform(lstm_features)
            lstm_sequence = lstm_features_scaled.reshape(1, 60, -1)
            
            predictions['lstm'] = self.models['lstm'].predict(lstm_sequence)[0][0]
        
        # Combine predictions using ensemble weights
        weighted_sum = sum(predictions[name] * ensemble_weights.get(name, 1.0) 
                          for name in predictions)
        total_weight = sum(ensemble_weights.get(name, 1.0) for name in predictions)
        
        ensemble_prediction = weighted_sum / total_weight
        
        return {
            'ensemble': ensemble_prediction,
            'individual': predictions,
            'weights_used': ensemble_weights
        }
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        importance_data = {}
        
        if 'rf' in self.models:
            rf_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.models['rf'].feature_importances_
            }).sort_values('importance', ascending=False)
            importance_data['random_forest'] = rf_importance
        
        if 'gb' in self.models:
            gb_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.models['gb'].feature_importances_
            }).sort_values('importance', ascending=False)
            importance_data['gradient_boosting'] = gb_importance
        
        return importance_data
    
    def plot_predictions(self, test_data, predictions, title="Stock Price Predictions"):
        """Create visualization of predictions vs actual prices"""
        plt.figure(figsize=(15, 8))
        
        # Plot actual prices
        plt.plot(test_data.index, test_data['Close'], 
                label='Actual Price', color='black', linewidth=2)
        
        # Plot predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, pred_values) in enumerate(predictions.items()):
            if model_name != 'ensemble':
                plt.plot(test_data.index, pred_values, 
                        label=f'{model_name.upper()} Prediction', 
                        color=colors[i % len(colors)], alpha=0.7)
        
        # Plot ensemble prediction
        if 'ensemble' in predictions:
            plt.plot(test_data.index, predictions['ensemble'], 
                    label='Ensemble Prediction', color='red', 
                    linewidth=2, linestyle='--')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def run_backtest(predictor, data, start_date, end_date):
    """
    Run a backtest to see how well our model would have performed historically.
    
    This is crucial for evaluating trading strategies - we simulate what would
    have happened if we used our model to make trading decisions in the past.
    """
    print(f"Running backtest from {start_date} to {end_date}")
    
    backtest_data = data.loc[start_date:end_date].copy()
    
    # We'll simulate a simple trading strategy:
    # - Buy when model predicts price will go up more than 2%
    # - Sell when model predicts price will drop more than 2%
    # - Hold otherwise
    
    positions = []  # Track our trading positions
    cash = 10000    # Starting cash
    shares = 0      # Starting shares
    portfolio_values = []
    
    for i in range(60, len(backtest_data) - 1):  # Start after LSTM lookback period
        current_date = backtest_data.index[i]
        current_price = backtest_data['Close'].iloc[i]
        
        # Get recent data for prediction
        recent_data = data.loc[:current_date].iloc[-100:]  # Use last 100 days
        
        # Make prediction
        try:
            prediction_result = predictor.predict(recent_data)
            predicted_price = prediction_result['ensemble']
            
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price
            
            # Trading decision
            if expected_return > 0.02 and shares == 0:  # Buy signal
                shares = cash / current_price
                cash = 0
                positions.append(('BUY', current_date, current_price, shares))
                
            elif expected_return < -0.02 and shares > 0:  # Sell signal
                cash = shares * current_price
                positions.append(('SELL', current_date, current_price, cash))
                shares = 0
            
            # Calculate portfolio value
            portfolio_value = cash + shares * current_price
            portfolio_values.append({
                'Date': current_date,
                'Portfolio_Value': portfolio_value,
                'Stock_Price': current_price,
                'Position': 'LONG' if shares > 0 else 'CASH'
            })
            
        except Exception as e:
            print(f"Error on {current_date}: {e}")
            continue
    
    # Convert to DataFrame for analysis
    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df.set_index('Date', inplace=True)
    
    # Calculate performance metrics
    total_return = (portfolio_df['Portfolio_Value'].iloc[-1] / 10000 - 1) * 100
    buy_hold_return = (backtest_data['Close'].iloc[-1] / backtest_data['Close'].iloc[60] - 1) * 100
    
    print(f"\nBacktest Results:")
    print(f"Strategy Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Excess Return: {total_return - buy_hold_return:.2f}%")
    print(f"Number of Trades: {len(positions)}")
    
    return portfolio_df, positions

# Example usage and demonstration
def main():
    """
    Main function to demonstrate the stock predictor.
    
    This shows the complete workflow:
    1. Generate/load data
    2. Train models
    3. Make predictions
    4. Evaluate performance
    5. Run backtests
    """
    print("=" * 60)
    print("ADVANCED STOCK PRICE PREDICTOR DEMO")
    print("=" * 60)
    print()
    
    # Step 1: Generate sample data
    print("Step 1: Loading stock data...")
    stock_data = generate_sample_data("AAPL", days=800)
    
    # Display basic info about our data
    print(f"Data range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
    print(f"Starting price: ${stock_data['Close'].iloc[0]:.2f}")
    print(f"Ending price: ${stock_data['Close'].iloc[-1]:.2f}")
    print(f"Total return: {(stock_data['Close'].iloc[-1]/stock_data['Close'].iloc[0] - 1)*100:.2f}%")
    print()
    
    # Step 2: Initialize and train predictor
    print("Step 2: Training prediction models...")
    predictor = StockPredictor(target_days=1)
    predictor.prepare_data(stock_data)
    predictor.train_models()
    print()
    
    # Step 3: Make some predictions
    print("Step 3: Making predictions...")
    
    # Use last 100 days to predict next day
    recent_data = stock_data.iloc[-100:]
    prediction_result = predictor.predict(recent_data)
    
    current_price = stock_data['Close'].iloc[-1]
    predicted_price = prediction_result['ensemble']
    expected_return = (predicted_price - current_price) / current_price * 100
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price (1 day): ${predicted_price:.2f}")
    print(f"Expected Return: {expected_return:.2f}%")
    print()
    
    print("Individual Model Predictions:")
    for model_name, pred_value in prediction_result['individual'].items():
        model_return = (pred_value - current_price) / current_price * 100
        print(f"  {model_name.upper():>15}: ${pred_value:.2f} ({model_return:+.2f}%)")
    print()
    
    # Step 4: Show feature importance
    print("Step 4: Feature Importance Analysis...")
    importance_data = predictor.get_feature_importance()
    
    if 'random_forest' in importance_data:
        print("Top 10 Most Important Features (Random Forest):")
        top_features = importance_data['random_forest'].head(10)
        for idx, row in top_features.iterrows():
            print(f"  {row['feature']:>25}: {row['importance']:.4f}")
    print()
    
    # Step 5: Run a simple backtest
    print("Step 5: Running backtest...")
    
    # Use middle portion of data for backtest
    backtest_start = stock_data.index[400]
    backtest_end = stock_data.index[700]
    
    try:
        portfolio_df, trades = run_backtest(predictor, stock_data, backtest_start, backtest_end)
        
        print(f"\nTrade History:")
        for trade in trades[-5:]:  # Show last 5 trades
            action, date, price, amount = trade
            if action == 'BUY':
                print(f"  {date.date()}: {action} {amount:.2f} shares at ${price:.2f}")
            else:
                print(f"  {date.date()}: {action} for ${amount:.2f} total")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
    
    print()
    print("=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print()
    print("IMPORTANT DISCLAIMERS:")
    print("- This is for educational purposes only")
    print("- Past performance doesn't guarantee future results")
