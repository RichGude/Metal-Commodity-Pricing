'''
Author:     Rich Gude
Purpose:    To amend metal commodity pricing data for time-series analysis
Revision:   1, dated April 1, 2021

The following code follows and borrows extensively from the TensorFlow homepage tutorial for Time-Series Structured Data
  modeling ('https://www.tensorflow.org/tutorials/structured_data/time_series').  I have ammended as appropriate for the 
  price data within this project and commented to better understand the flow of the work.

Document Review: This file loads economic data procured via .xlsx format from the office of Amit Goyal at his personal
  site (http://www.hec.unil.ch/agoyal) published in conjunction with his paper, "A Comprehensive Look at the Empirical
  Performance of Equity Premium Prediction".  The paper reviews the lagged effect multiple economic factors have equity
  premiums.  Specifically, this paper found that the numerous variables had the most significant correlation and
  suitability for forecasting equity premium trends over a lagged time frame.  Wang et al. (2020) utilized eight of
  these variables in their paper with limited forecasting success.  The following variables will be used for predictive
  Recurring Neural Network (RNN) creation:

  The following fourteen (14) variables are collected monthly from January 1990 to December 2019, relatively matching
  the commodity price data collected from the IMF:
  - Dividends (D12): Dividends are twelve-month moving sums of dividends paid on the S&P 500 index; data obtained from
    the S&P Corporation.
  - Earnings (E12): Earnings are twelve-month moving sums of earnings on the S&P 500 index. Data obtained from Robert
    Shiller’s website for the period 1990 to June 2003. Earnings from June 2003 to 2020 are from Goyal estimates on
    interpolation of quarterly earnings provided by S&P Corporation.
  - Book to Market Ratio (b/m): the ratio of book value to market value for the Dow Jones Industrial Average. For the
    months of March to December, computed by dividing book value at the end of previous year by the price at the end of
    the current month. For the months of January to February, this is computed by dividing book value at the end of 2
    years ago by the price at the end of the current month.
  - Treasury Bills (tbl):  T-bill rates from 1990 to 2020 are the 3-Month Treasury Bill: Secondary Market Rate.
  - Corporate Bond Yields (AAA): Yields on AAA-rated for the period 1990 to 2020
  - Corporate Bond Yields (BAA): Yields on BAA-rated bonds for the period 1990 to 2020
  - Long Term Yield (lty): Long-term government bond yields for the period 1990 to 2020
  - Net Equity Expansion (ntis): the ratio of twelve-month moving sums of net issues by NYSE listed stocks divided by
    the total market capitalization of NYSE stocks. This dollar amount of net equity issuing activity (IPOs, SEOs, stock
    repurchases, less dividends) for NYSE listed stocks is computed from Center for Research in Security Prices data
  - Risk-free Rate (Rfree): The risk-free rate for the period 1990 to 2020 is the T-bill rate
  - Inflation (infl): the Consumer Price Index (All Urban Consumers) for the period 1990 to 2020 from the Bureau of
    Labor Statistics, lagged by one month to account for distribution lag.
  - Long Term Rate of Return (ltr): Long-term government bond returns for the period 1990 to 2020 are from Ibbotson’s
    Stocks, Bonds, Bills and Inflation Yearbook.
  - Corporate Bond Returns (corpr): Long-term corporate bond returns for the period 1990 to 2020 are from Ibbotson’s
    Stocks, Bonds, Bills and Inflation Yearbook.
  - Stock Variance (svar): Stock Variance is computed as sum of squared daily returns on S&P 500. Daily returns from
    1990 to 2020 are obtained from CRSP.
  - Stock Prices (SPvw): S&P 500 index prices from 1990 to 2020 from CRSP’s month-end values. Stock Returns are the
    continuously-compounded returns on the S&P 500 index.

'''

# %% Prep library and model constants

import os                   # for specifying working directory commands
import openpyxl             # for appending to excel files
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt     # for graph illustration
import numpy as np                  # for pandas value typing
import pandas as pd                 # for csv file reading and dataFrame manipulation
import seaborn as sns               # for specific graphics
import tensorflow as tf
# for splitting training and test data
from sklearn.model_selection import train_test_split

# %% Data Preprocessing and Loading

# Choose a metal to evaluate
metals = ['Aluminum']

# Define figure constants:
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Define other constants:
# set data working directory
dwd = os.path.join(os.getcwd(), 'EconData')
metal_list = ['Aluminum', 'Copper', 'IronOre',
              'Nickel', 'Zinc']      # set list of metal names
var_names = ['D12', 'E12', 'b/m', 'tbl', 'AAA', 'BAA', 'lty',
             'ntis', 'Rfree', 'infl', 'ltr', 'corpr', 'svar', 'SPvw']

# %# Load data and begin data preparation:
econData = pd.read_excel(os.path.join(dwd, 'PredictorData2019.xlsx'), sheet_name='Monthly',
                         usecols=['Date'] + var_names, index_col=0)
priceData = pd.read_excel(os.path.join(
    dwd, 'PriceData.xlsx'), sheet_name='1990Price', index_col=0)

# Review Data
# print("Economic Indicators:\n', econData.head())
# print("Commodity Price Values:\n', priceData.head())
# showCase = econData.plot(subplots=True)
# plt.show()
# All data looks acceptable and is otherwise able to proceed:

# %# Train/Test/Validate Split
# data_length = len(econData)
# train_econ = econData[0:int(data_length*0.7)]
# valid_econ = econData[int(data_length*0.7):int(data_length*0.9)]
# test_econ = econData[int(data_length*0.9):]
#
# train_price = priceData[0:int(data_length*0.7)]
# valid_price = priceData[int(data_length*0.7):int(data_length*0.9)]
# test_price = priceData[int(data_length*0.9):]
#
# # Normalize Economic data
# econ_mean = train_econ.mean()
# econ_stdv = train_econ.std()
# train_econ = (train_econ - econ_mean)/econ_stdv
# valid_econ = (valid_econ - econ_mean)/econ_stdv
# test_econ = (test_econ - econ_mean)/econ_stdv
#
# # Normalize Price data
# price_mean = train_price.mean()
# price_stdv = train_price.std()
# train_price = (train_price - price_mean)/price_stdv
# valid_price = (valid_price - price_mean)/price_stdv
# test_price = (test_price - price_mean)/price_stdv

# Review normalized data structure in a violin plot
# sample_std = (econData - econData.mean()) / econData.std()      # Define a new variable so as to not adjust current
# # Create a reverse-pivot table basically from a two-column dataframe for each violin-chart creation
# sample_std = sample_std.melt(var_name='Variable', value_name='Normalized')
# plt.figure(figsize=(14, 8))
# ax = sns.violinplot(x='Variable', y='Normalized', data=sample_std)
# _ = ax.set_xticklabels(econData.keys(), rotation=90)
# plt.show()
# All data looks acceptable and is otherwise able to proceed (despite large outliers on 'svar' and 'infl')

# Create a window for reviewing past economic variable data for predicting current commodity prices
'''
In some cases, economic data may have a absolute and immediate impact on price data, such as inflation, discussed in the
Time Series section and quantitatively, but not qualitatively excluded from the real price data here, or treasury bill
rates, which are published by the government and can be immediately assessed for their return on investment over various
other investment opportunities.  In other cases, this data may have a delayed effect on price data, such as corporate
bond return or stock variance data being compiled and released for investor consumption and integration into buying and
selling behavior at a later date from their real-world calculation.

For this delayed consumption effect reason, a six-month window going back in time from the current day will be used to
predict commodity price for the next month  (e.g., using economic data from January through June, calendar months 1
through 6, to predict commodity prices in July, calendar month 7).

Define a class that takes in economic factors and price data dataframes
'''

# Define a class that takes in economic factors and price data dataframes


class SampleGenerator:
    def __init__(self, metal_label, input_width=6, label_width=1, shift=1,
                 econ_data=econData, comm_data=priceData):

        # Concatenate the metal label into the economic data to make one dataset from which to pull data
        self.data = pd.concat([econ_data, comm_data[metal_label]], axis=1)

        # Split (70:20:10, train/validation/test):
        self.data_length = len(self.data)
        self.trn_data = self.data[0:int(self.data_length * 0.7)]
        self.val_data = self.data[int(
            self.data_length * 0.7):int(self.data_length * 0.9)]
        self.tst_data = self.data[int(self.data_length * 0.9):]

        # Normalize the data:
        # Must use mean and standard deviation of training data, for appropriate rigor
        self.data_mean = self.trn_data.mean()
        self.data_stdv = self.trn_data.std()
        self.trn_data = (self.trn_data - self.data_mean) / self.data_stdv
        self.val_data = (self.val_data - self.data_mean) / self.data_stdv
        self.tst_data = (self.tst_data - self.data_mean) / self.data_stdv

        # Work out the label column indices.
        # metal_label but be a list with string name(s) of metals(s) from list
        self.metal_label = metal_label
        self.column_indices = {name: i for i, name in
                               enumerate(self.trn_data.columns)}

        # Work out the window parameters (input and label widths are
        # standard is 6 (i.e., 6 months back of information)
        self.input_width = input_width
        # standard is 1 (i.e., 1 month of prediction)
        self.label_width = label_width
        # standard is 1 (i.e., 1 month forward in prediction)
        self.shift = shift

        # standard is 6 back + 1 forward = *7*
        self.total_window_size = self.input_width + self.shift

        # standard is 'slice(0, 6, None)'
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]    # std is 'array([0, 1, 2, 3, 4, 5])'

        self.label_start = self.total_window_size - \
            self.label_width                # standard is 7 - 1 = *6*
        # standard is 'slice(6, None, None)'
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]   # standard is 'array([6])'

    # Define output for self-calling a SampleGenerator object
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name: {self.metal_label}'])

    # SampleGenerator instance has a single object with all feature and label data.  Create a function, 'split_window',
    #   to separate single instance into two objects of features and labels over the same time frame
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        labels = tf.stack([labels[:, :, self.column_indices[name]]
                           for name in self.metal_label], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the 'tf.data.Datasets' are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # Using a time-series DataFrame object, convert to TensorFlow data.Dataset object in feature and label window pairs
    def make_dataset(self, data, batch=6):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            # numpy array containing consecutive-time data points
            data=data,
            targets=None,                               # set to 'None' to only yield input data
            # number of time steps in output sequence (std is *7*)
            sequence_length=self.total_window_size,
            # How many time steps to skip between each batch
            sequence_stride=1,
            # shuffle output sequences to improve model rigor
            shuffle=True,
            batch_size=batch, )                         # set batch size of Dataset (std is *6*)

        # Automatically separate data into feature and label sets
        ds = ds.map(self.split_window)

        return ds

    # Define property values for training, validating, and testing data
    @property
    def train(self):
        return self.make_dataset(self.trn_data)

    @property
    def validate(self):
        return self.make_dataset(self.val_data)

    @property
    def test(self):
        return self.make_dataset(self.tst_data)
    # Use [object].[Dataset_function].element_spec to review the structure of the Dataset
    # e.g.: std_window.train.element_spec =
    #   (TensorSpec(shape=(None, 6, 15), dtype=tf.float32, name=None),
    #    TensorSpec(shape=(None, 1, 1), dtype=tf.float32, name=None))

    @property
    def example(self):
        # Get and cache an example batch of `inputs, labels` for plotting and asset investigation
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the '.train' dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    # Construct a function for viewing model outputs:
    def plot(self, model=None, plot_col=metals[0], max_subplots=3):
        # Pull a batch (std is *6*) of window values and save the input and label tensors
        inputs, labels = self.example
        # Generate a standard figure
        plt.figure(figsize=(12, 8))
        # Store the value of the label in the input column index (std is *14*)
        plot_col_index = self.column_indices[plot_col]
        # Plot subplots for each element in the batch (*6*) or max_sub, whichever is smaller
        max_n = min(max_subplots, len(inputs))
        # For each subplot:
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            # Each subplot will show real metal price
            plt.ylabel(f'{plot_col} Price [normed]')
            # Plot the price values for each of the training time steps (i.e., non-forecasted)
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            # The label index for a single list metal_label name is always 0
            if self.metal_label:
                label_col_index = 0
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            # If there is a label for the window, plot the labels (the actual values for each forecast)
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            # 'plot' works without a model (will ust show input and label prices); if there is a model, plot
            #  the predicted values for comparison with the label values (which will share an x-axis value)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Month')


# %% Run Generic SampleGenerator (6 months back to predict one month forward)


# Define a standard model window (1-month-ahead prediction from data up to six months behind)
std_window = SampleGenerator(metal_label=metals)
# Define a forecasting model window (6-month-ahead prediction from data up to six-months behind)
ahead_window = SampleGenerator(metal_label=metals, label_width=6, shift=6)
# Define appropriate window for baseline model (1 month back to predict one month forward)
single_window = SampleGenerator(metal_label=metals, input_width=1)

# Display and confirm batch and input/label sizes
for example_inputs, example_labels in std_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
column_names = pd.concat([econData, priceData[metals]], axis=1).columns
column_indices = {name: i for i, name in enumerate(column_names)}


# For ease in testing models later, define a function for testing separate models on separate windows
MAX_EPOCHS = 50


def compile_and_fit(model, window, patience=5):
    # Stop the model compiling if the value-loss parameter doesn't decrease at least once over two consecutive cycles
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',   # monitor validation loss, vice training
                                                      patience=patience,
                                                      mode='min')
    # Compile model with standard loss and optimizer values
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    # Save weights only if the val_loss improves (decreases) and load those weights after model creation
    checkpoint_filepath = '/tmp/check_weights'
    model_weight_saving = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    # Output model fit data for storing and comparing
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.validate,
                        callbacks=[early_stopping, model_weight_saving])
    # load best weights
    model.load_weights(checkpoint_filepath)
    return history


'''
The Time-Series Analysis (ARIMA modeling) previously performed, despite not otherwise having robust prediction quality,
provides insight into the underlying trend of the price data; that is, for each commodity, the principal trend is a
random walk.  Essentially, the commodity price for any particular month is the price from the previous month, altered
by a random fluctuation in with an experimentally-derived mean (not statistically different from 0) and standard
deviation.  Below is a baseline prediction algorithm that models this prediction behavior by predicting that the next
month's commodity price will be the previous month's commodity price; this is the baseline model.
'''

# Define Baseline Model (need a special subclass of keras.Model)


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        # declare Baseline as a subclass of the tf.keras.Model class, inheriting functionality
        super().__init__()
        self.label_index = label_index

    # '__call__' is a generic function for whenever you reference just the class name
    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]     # result.shape = (None, 1)
        # returnTensor.shape = (None, 1, 1)
        return result[:, :, tf.newaxis]


'''
Machine Learning models with multiple layers can be complex to the point where the interactions between variables and
their calculated weights are no longer understandable to even the experienced machine learning programmer.  This effect 
can be positive since it allows more robust (and potentially overfit) predicted values better matching the actual values;
however, it does not aid in a simplistic understanding of each factor's role within the model.  The simplest model that
can be built is a linear regression model; this model is simple enough that extracting the weights for each factor at 
each time step shows the correlated effect that factor has on the predicted price value.
'''
# Define a simple Linear model (this is used for discussing the role of each factor on the predicted price w/ 1 ts)
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

'''
The documentation for the tf.keras.layer.dense class identifies that a 3-rank tensor fed into the layer will be 'shrunk',
in a sense, into a 2-rank tensor via the computation of the dot product between the inputs and the kernel along the last
axis.  This means that a (6,6,15) input tensor fed into the single-Dense-layer model will return the weight for all the
time layers agglomerated together.  In order to get a single 2-D matrix showing the weights for each input at each time 
a Flattened and reshaped model is needed.  The tf.keras Flatten() function lays out all time layers with their input 
factors sequentially from 6-months past to present.  The final output will be reshaped accordingly to place the correct 
numerical weight with each time and input.
'''
# Define a larger Linear model (this is used for discussing the role of each factor on the predicted price w/ 6 ts)
linMulti = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1),
    tf.keras.layers.Reshape([1, -1])
])

# Define a Recurring Neural Network (RNN) model using the Long Short Term Memory (LSTM) Layer
lstm_model = tf.keras.models.Sequential([
    # Shape [6, 6, 15] => [6, 6, 30], because return_sequences is False, model does not predict for each time step
    tf.keras.layers.LSTM(30, return_sequences=False),
    # Shape => [6, 1, 1]
    tf.keras.layers.Dense(units=1)
])

# Define a Recurring Neural Network (RNN) model for predicting up to 6 months out
lstm_ahead_model = tf.keras.models.Sequential([
    # Shape [6, 6, 15] => [6, 30], because return_sequences is true, model predicts for each time step
    tf.keras.layers.LSTM(30, return_sequences=False),
    # Shape => [6, 90]
    tf.keras.layers.Dense(
        units=6*15, kernel_initializer=tf.initializers.zeros()),
    # Shape => [6, 6, 15]
    tf.keras.layers.Reshape([6, 15])
])

# Initiate value and performance dictionary to compare future models with baseline
val_performance = {}
performance = {}

# %% Create a Baseline model and store results

# Evaluate the Baseline models with TensorFlow/Keras performance indicators
basePricePred = Baseline(label_index=column_indices['Aluminum'])

basePricePred.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

# Save value
val_performance['Baseline'] = basePricePred.evaluate(single_window.validate)
performance['Baseline'] = basePricePred.evaluate(single_window.test, verbose=0)

# %% Create a Linear model and store results
history = compile_and_fit(linear, single_window)
# Save weight outputs as an Excel file (kernel is the weights matrix taken from the first/only layer)
single_weights = pd.Series(linear.layers[0].kernel[:, 0].numpy(),
                           index=column_names)
single_weights.to_excel(os.path.join(dwd, metals[0] + '_sing_weights.xlsx'))

val_performance['Linear'] = linear.evaluate(single_window.validate)
performance['Linear'] = linear.evaluate(single_window.test, verbose=0)

# # %% Create a Larger Linear model and store results
# history = compile_and_fit(linMulti, std_window)
# # Save weight outputs as an Excel file
# multi_weights = pd.DataFrame(linMulti.layers[1].kernel[:, 0].numpy().reshape((6, -1)),
#                              columns=column_names, index=[-6, -5, -4, -3, -2, -1])
# multi_weights.to_excel(os.path.join(dwd, metals[0] + '_multi_weights.xlsx'))

# val_performance['LinMulti'] = linMulti.evaluate(std_window.validate)
# performance['LinMulti'] = linMulti.evaluate(std_window.test, verbose=0)

# # %% Create a 1-Step-Ahead RNN Network Model and store results
# history = compile_and_fit(lstm_model, std_window)

# val_performance['RNN-1'] = lstm_model.evaluate(std_window.validate)
# performance['RNN-1'] = lstm_model.evaluate(std_window.test, verbose=0)

# # %% Create a 1-Step-Ahead RNN Network Model and store results
# history = compile_and_fit(lstm_ahead_model, ahead_window)
# print('6-Step RNN Summary:\n', lstm_ahead_model.summary)

# val_performance['RNN-6'] = lstm_ahead_model.evaluate(ahead_window.validate)
# performance['RNN-6'] = lstm_ahead_model.evaluate(ahead_window.test, verbose=0)
