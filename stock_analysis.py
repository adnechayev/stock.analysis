import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_ta as ta
import warnings

warnings.filterwarnings("ignore")


class TeslaStock():
    def __init__(self, df):
        self.df = df

    def eda(self):
        """
        Perform some exploratory data analysis.

        return: first 5 rows, column data types, number of duplicates, and number of null values.
        """
        head = self.df.head()
        info = self.df.info()
        dupes = self.df.duplicated().sum()
        na = self.df.isna().sum()
        
        output = f"{head}\n\n{info}\n\nNumber of duplicates: {dupes}\n\nNumber of NaN values:\n{na}"
        
        print(output)
    
    def filter_df_date(self, date_col):
        """
        Adjust the date column to only show stock prices for the year of 2023.

        param: date column
        return: new dataframe with adjusted date column
        """
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        start_date = '2023-01-01'
        end_date = '2024-01-01'
        
        filtered_df = self.df[(self.df[date_col] >= start_date) & (self.df[date_col] <= end_date)]

        return filtered_df
    
    def line_plot(self, df, adj_close, date):
        """
        Creates a line plot showcasing the adjusted closing price of the stock over the course of the year.

        param1: dataframe
        param2: adjusted closing price column
        param3: date column
        return: line plot
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=date, y=adj_close, data=df)
        plt.title('Stock Price Trend')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Closing Price')
        plt.show()
    
    def calculate_percentage_change(self, df, adj_close):
        """
        Calculate the percentage change of the stock price over the course of the year.

        param1: dataframe
        param2: adjusted closing price column
        return: lowest stock price, highest stock price, percentage change
        """
        min_value = df[adj_close].min()
        max_value = df[adj_close].max()
        percentage_change = ((max_value - min_value) / min_value) * 100

        return min_value, max_value, percentage_change
    
    def tech_analysis(self, df, adj_close):
        """
        Create and calculate an EMA (Exponential Moving Average), and append to dataset.

        param1: dataframe
        param2: adjusted closing price column
        return: dataframe with added EMA column
        """
        df["ema"] = ta.ema(close = adj_close, length = 10)
        df.dropna(subset=["ema"], inplace=True)

        return df
    
    def ema_plot(self, df, adj_close, date, ema_col, ema_length):
        """
        Create a double line plot showing the adjusted closing price trend and the EMA trend side by side.

        param1: dataframe
        param2: adjusted closing price column
        param3: date column
        param4: EMA column
        param5: number of periods over which EMA is calculated
        return: line plot
        """
        plt.figure(figsize=(10, 6))
    
        sns.lineplot(x=date, y=adj_close, data=df, label='Adjusted Closing Price')
    
        sns.lineplot(x=date, y=ema_col, data=df, label=f'EMA ({ema_length} periods)')

        plt.title('Stock Price Trend with EMA')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def model_metrics(self, model_coef, y_test, y_pred):
        """
        Print the linear regression model coeffecients, mean absolute error, and coefficient of determination

        param1: model coefficients
        param2: test values
        param3: predicted values
        return: model coeffecients, mean absolute error, and coefficient of determination
        """
        print(f"Model Coefficients: {model_coef} | "
        f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f} | "
        f"Coefficient of Determination: {r2_score(y_test, y_pred):.4f}")

    def predict_future_prices(self, model, start_date, periods, adj_close_col):
        """
        Predict future stock prices using the trained model.

        param1: trained linear regression model
        param2: start date for predictions
        param3: number of periods (days) to predict
        param4: adjusted closing price column
        return: DataFrame with future dates and predicted prices
        """
        future_dates = pd.date_range(start=start_date, periods=periods, freq='D')
        last_adj_close = self.df[adj_close_col].iloc[-1]

        future_df = pd.DataFrame(future_dates, columns=['Date'])
        future_df[adj_close_col] = np.nan  # Initialize with NaN

        # Iteratively predict future prices
        for i in range(len(future_df)):
            if i == 0:
                future_df.at[i, adj_close_col] = last_adj_close
            else:
                future_df.at[i, adj_close_col] = model.predict(np.array([[future_df.at[i-1, adj_close_col]]]))

        return future_df

    def plot_predictions(self, filtered_df, future_df, adj_close_col):
        """
        Plot actual and predicted stock prices.

        param1: filtered DataFrame with actual prices
        param2: DataFrame with future dates and predicted prices
        param3: adjusted closing price column
        return: line plot
        """
        plt.figure(figsize=(10, 6))

        sns.lineplot(x=filtered_df['Date'], y=filtered_df[adj_close_col], label='Actual Prices')
        sns.lineplot(x=future_df['Date'], y=future_df[adj_close_col], label='Predicted Prices')

        plt.title('Actual and Predicted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.show()


        
    

df = pd.read_csv("Tesla Dataset.csv")

# Running exploratory data analysis on dataset
tesla_stock = TeslaStock(df)
tesla_stock.eda()

# Adjusting the Date column to standard date-time format, and only include year 2023
filtered_df = tesla_stock.filter_df_date("Date")
plot = tesla_stock.line_plot(filtered_df, "Adj Close", "Date")

# Calculating percentage change in stock price
min_value, max_value, percentage_change = tesla_stock.calculate_percentage_change(filtered_df, "Adj Close")
print(f"Minimum value: {min_value}")
print(f"Maximum value: {max_value}")
print(f"Percentage change: {percentage_change:.2f}%")

# Append EMA values to dataframe and plot out EMA trend
ema = tesla_stock.tech_analysis(df, df["Adj Close"])
ema_plot = tesla_stock.ema_plot(ema, 'Adj Close', 'Date', 'ema', ema_length=10)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(ema[['Adj Close']], ema[['ema']], test_size=.2)

from sklearn.linear_model import LinearRegression

# Building and running the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Printing MSE, R2 Score, and MAE
tesla_stock.model_metrics(model.coef_, y_test, y_pred)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted Values')

# Plot the ideal line (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Line (y = x)')

plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Predict future stock prices
future_df = tesla_stock.predict_future_prices(model, '2024-01-02', 30, 'Adj Close')

# Plot actual and predicted stock prices
tesla_stock.plot_predictions(filtered_df, future_df, 'Adj Close')

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(ema[['Adj Close', 'Volume', 'ema']])

# Prepare the data for training
X = []
y = []

window_size = 60  # Look back 60 days
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for the neural network
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the neural network model
model = Sequential()

# Adding the first LSTM layer and some Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularization
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the RNN
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, 0]

# Plot the results
zoom_range = 50 
plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(np.concatenate((y_test[:zoom_range].reshape(-1, 1), np.zeros((y_test[:zoom_range].shape[0], 2))), axis=1))[:, 0], color='blue', label='Actual')
plt.plot(predictions[:zoom_range], color='red', label='Predicted', alpha=0.6)
plt.title('Stock Price Prediction (Zoomed-In)')
plt.xlabel('Sample')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Predict future prices
def predict_future_prices_rnn(model, scaled_data, window_size, days_to_predict):
    future_prices = []

    last_window = scaled_data[-window_size:, 0]  # Take the last window_size days from the data

    for _ in range(days_to_predict):
        input_data = last_window.reshape((1, window_size, 1))
        predicted_price = model.predict(input_data)[0, 0]
        future_prices.append(predicted_price)

        # Update the last_window by removing the first element and adding the predicted price
        last_window = np.append(last_window[1:], predicted_price)

    return future_prices

# Number of future days to predict
days_to_predict = 30

# Predict future prices
future_predictions = predict_future_prices_rnn(model, scaled_data, window_size, days_to_predict)

# Transform the predictions back to the original scale
future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((len(future_predictions), 2))), axis=1))[:, 0]

# Create future dates for plotting
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict, freq='D')

# Plot the historical and future predictions
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Adj Close'], label='Historical Prices')
plt.plot(future_dates, future_predictions, label='Predicted Future Prices', linestyle='--', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()






