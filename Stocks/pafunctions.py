
# Function to calculate the Relative Strength Index (RSI)
def calculate_rsi(data, window):
    # Get the difference in price from the previous step
    delta = data['Close'].diff()
    delta = delta[1:]  # Remove the first NA value

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA (Exponential Weighted Moving Average)
    roll_up1 = up.ewm(span=window).mean()
    roll_down1 = down.abs().ewm(span=window).mean()

    # Calculate the RSI based on EWMA
    rs = roll_up1 / roll_down1
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi

# Function to calculate the Moving Average Convergence Divergence (MACD)
def calculate_macd(data, slow, fast, signal):
    # Calculate the Short Term Exponential Moving Average
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    # Calculate the Long Term Exponential Moving Average
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    # Calculate the MACD Line
    macd = ema_fast - ema_slow
    # Calculate the Signal Line
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    
    return macd, macd_signal


# Function to calculate VWAP (assumes df has 'Volume' and 'Close' columns)
def calculate_vwap(data):
    vwap = (data['Volume'] * data['Close']).cumsum() / data['Volume'].cumsum()
    return vwap

# Function to calculate Momentum
def calculate_momentum(data, n):
    momentum = data['Close'].diff(n)
    return momentum

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window, num_of_std):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band


# Function to calculate Pivot Points
def calculate_pivot_points(data):
    pivot_points = {}
    pivot_points['P'] = (data['High'] + data['Low'] + data['Close']) / 3.0

    # Support levels
    pivot_points['S1'] = (pivot_points['P'] * 2) - data['High']
    pivot_points['S2'] = pivot_points['P'] - (data['High'] - data['Low'])
    pivot_points['S3'] = data['Low'] - 2 * (data['High'] - pivot_points['P'])

    # Resistance levels
    pivot_points['R1'] = (pivot_points['P'] * 2) - data['Low']
    pivot_points['R2'] = pivot_points['P'] + (data['High'] - data['Low'])
    pivot_points['R3'] = data['High'] + 2 * (pivot_points['P'] - data['Low'])

    return pivot_points

# Function to calculate Fibonacci Retracements
def calculate_fibonacci_retracements(high, low):
    # Fibonacci levels
    levels = [0.0, 23.6, 38.2, 50.0, 61.8, 100.0]
    
    diff = high - low
    retracements = {}
    for level in levels:
        retracements[f'Fib_{level}'] = high - (level / 100.0) * diff

    return retracements