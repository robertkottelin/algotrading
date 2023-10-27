import pandas as pd
import yfinance as yf
from fredapi import Fred

# Set the starting point for all data
start_date = '1990-01-01'

# Initialize the Fred instance with your API key
fred = Fred(api_key='439fddd0f1a7c4e91f765bbcf07dcc74')  # make sure to use your API key

# Fetch economic data with the specified start date
gdp = fred.get_series('GDP', observation_start=start_date)
unemployment = fred.get_series('UNRATE', observation_start=start_date)
cpi = fred.get_series('CPIAUCSL', observation_start=start_date)

# Convert the economic data to DataFrames and reset their indexes
gdp = gdp.reset_index()
unemployment = unemployment.reset_index()
cpi = cpi.reset_index()

# Rename the 'index' column to 'Date' for merging
gdp.rename(columns={'index': 'Date', 0: 'GDP'}, inplace=True)
unemployment.rename(columns={'index': 'Date', 0: 'Unemployment'}, inplace=True)
cpi.rename(columns={'index': 'Date', 0: 'CPI'}, inplace=True)

# Fetch S&P 500 data
tickerSymbol = '^GSPC'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(start=start_date)  # use the same start date
tickerDf.index = tickerDf.index.tz_localize(None)
# drop the dividends and stock splits columns
tickerDf.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

# # Fetch VIX data
vix_data = yf.download('^VIX', start=start_date)  # use the same start date
vix_data.reset_index(inplace=True)  # VIX data needs a 'Date' column for merging

merged_df = pd.merge(tickerDf, vix_data[['Date', 'Close']], on='Date', how='outer', suffixes=('', '_VIX'))
merged_df = pd.merge(merged_df, gdp, on='Date', how='outer')
merged_df = pd.merge(merged_df, unemployment, on='Date', how='outer')
merged_df = pd.merge(merged_df, cpi, on='Date', how='outer')

print("Merged DataFrame: ", merged_df.head())

# At this point, you'll have a DataFrame with potentially many missing values, especially in the economic columns
# that originally had a lower frequency (e.g., quarterly for GDP). You must decide how to handle these missing values.

# A common approach is to forward-fill missing values for each column. This approach assumes that the last known value
# is still valid until a new value is known (as is the case with GDP, for example).

# merged_df.fillna(method='ffill', inplace=True)

# # Now, your DataFrame 'merged_df' contains S&P 500, VIX, GDP, unemployment, and CPI data, all starting from 1990.
# # It's crucial to inspect the final DataFrame to ensure the merging was conducted correctly and the data is consistent.

# # Print the head of the DataFrame to check the data
# print(merged_df.head())

# # Also, review the tail of the DataFrame to understand the data's current structure and contents
# print(merged_df.tail())


# # Support and Resistance FUNCTIONS
# def support(df1, l, n1, n2): #n1 n2 before and after candle l
#     for i in range(l-n1+1, l+1):
#         if(df1.Low[i]>df1.Low[i-1]):
#             return 0
#     for i in range(l+1,l+n2+1):
#         if(df1.Low[i]<df1.Low[i-1]):
#             return 0
#     return 1

# def resistance(df1, l, n1, n2): #n1 n2 before and after candle l
#     for i in range(l-n1+1, l+1):
#         if(df1.High[i]<df1.High[i-1]):
#             return 0
#     for i in range(l+1,l+n2+1):
#         if(df1.High[i]>df1.High[i-1]):
#             return 0
#     return 1

# length = len(df)
# High = list(df['High'])
# Low = list(df['Low'])
# Close = list(df['Close'])
# Open = list(df['Open'])
# bodydiff = [0] * length

# Highdiff = [0] * length
# Lowdiff = [0] * length
# ratio1 = [0] * length
# ratio2 = [0] * length

# def isEngulfing(l):
#     row=l
#     bodydiff[row] = abs(Open[row]-Close[row])
#     if bodydiff[row]<0.000001:
#         bodydiff[row]=0.000001      

#     bodydiffmin = 0.002
#     if (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
#         Open[row-1]<Close[row-1] and
#         Open[row]>Close[row] and 
#         (Open[row]-Close[row-1])>=-0e-5 and Close[row]<Open[row-1]): #+0e-5 -5e-5
#         return 1

#     elif(bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
#         Open[row-1]>Close[row-1] and
#         Open[row]<Close[row] and 
#         (Open[row]-Close[row-1])<=+0e-5 and Close[row]>Open[row-1]):#-0e-5 +5e-5
#         return 2
#     else:
#         return 0
       
# def isStar(l):
#     bodydiffmin = 0.0020
#     row=l
#     Highdiff[row] = High[row]-max(Open[row],Close[row])
#     Lowdiff[row] = min(Open[row],Close[row])-Low[row]
#     bodydiff[row] = abs(Open[row]-Close[row])
#     if bodydiff[row]<0.000001:
#         bodydiff[row]=0.000001
#     ratio1[row] = Highdiff[row]/bodydiff[row]
#     ratio2[row] = Lowdiff[row]/bodydiff[row]

#     if (ratio1[row]>1 and Lowdiff[row]<0.2*Highdiff[row] and bodydiff[row]>bodydiffmin):# and Open[row]>Close[row]):
#         return 1
#     elif (ratio2[row]>1 and Highdiff[row]<0.2*Lowdiff[row] and bodydiff[row]>bodydiffmin):# and Open[row]<Close[row]):
#         return 2
#     else:
#         return 0
    
# def CloseResistance(l,levels,lim):
#     if len(levels)==0:
#         return 0
#     c1 = abs(df.High[l]-min(levels, key=lambda x:abs(x-df.High[l])))<=lim
#     c2 = abs(max(df.Open[l],df.Close[l])-min(levels, key=lambda x:abs(x-df.High[l])))<=lim
#     c3 = min(df.Open[l],df.Close[l])<min(levels, key=lambda x:abs(x-df.High[l]))
#     c4 = df.Low[l]<min(levels, key=lambda x:abs(x-df.High[l]))
#     if( (c1 or c2) and c3 and c4 ):
#         return 1
#     else:
#         return 0
    
# def CloseSupport(l,levels,lim):
#     if len(levels)==0:
#         return 0
#     c1 = abs(df.Low[l]-min(levels, key=lambda x:abs(x-df.Low[l])))<=lim
#     c2 = abs(min(df.Open[l],df.Close[l])-min(levels, key=lambda x:abs(x-df.Low[l])))<=lim
#     c3 = max(df.Open[l],df.Close[l])>min(levels, key=lambda x:abs(x-df.Low[l]))
#     c4 = df.High[l]>min(levels, key=lambda x:abs(x-df.Low[l]))
#     if( (c1 or c2) and c3 and c4 ):
#         return 1
#     else:
#         return 0
# n1=2
# n2=2
# backCandles=30
# signal = [0] * length

# for row in range(backCandles, len(df)-n2):
#     ss = []
#     rr = []
#     for subrow in range(row-backCandles+n1, row+1):
#         if support(df, subrow, n1, n2):
#             ss.append(df.Low[subrow])
#         if resistance(df, subrow, n1, n2):
#             rr.append(df.High[subrow])
#     #!!!! parameters
#     if ((isEngulfing(row)==1 or isStar(row)==1) and CloseResistance(row, rr, 150e-5) ):#and df.RSI[row]<30
#         signal[row] = 1
#     elif((isEngulfing(row)==2 or isStar(row)==2) and CloseSupport(row, ss, 150e-5)):#and df.RSI[row]>70
#         signal[row] = 2
#     else:
#         signal[row] = 0


# df['signal']=signal
# df[df['signal']==1].count()
# df_subset = df[['Open', 'High', 'Low', 'Close', 'Volume']]
# print("New DataFrame with selected columns:")
# print(df_subset.head())
# #df=df.iloc[100:200]
# df
# def SIGNAL():
#     return df.signal

# from backtesting import Strategy

# class MyCandlesStrat(Strategy):  
#     def init(self):
#         super().init()
#         self.signal1 = self.I(SIGNAL)

#     def next(self):
#         super().next() 
#         if self.signal1==2:
#             sl1 = self.data.Close[-1] - 600e-4
#             tp1 = self.data.Close[-1] + 450e-4
#             self.buy(sl=sl1, tp=tp1)
#         elif self.signal1==1:
#             sl1 = self.data.Close[-1] + 600e-4
#             tp1 = self.data.Close[-1] - 450e-4
#             self.sell(sl=sl1, tp=tp1)

# from backtesting import Backtest

# bt = Backtest(df, MyCandlesStrat, cash=10_000, commission=.00)
# stat = bt.run()
# stat


# bt.plot()
# #Target flexible way
# pipdiff = 250*1e-4 #for TP
# SLTPRatio = 1 #pipdiff/Ratio gives SL
# def mytarget(barsupfront, df1):
#     length = len(df1)
#     High = list(df1['High'])
#     Low = list(df1['Low'])
#     Close = list(df1['Close'])
#     Open = list(df1['Open'])
#     trendcat = [None] * length
#     for line in range (0,length-barsupfront-2):
#         valueOpenLow = 0
#         valueOpenHigh = 0
#         for i in range(1,barsupfront+2):
#             value1 = Open[line+1]-Low[line+i]
#             value2 = Open[line+1]-High[line+i]
#             valueOpenLow = max(value1, valueOpenLow)
#             valueOpenHigh = min(value2, valueOpenHigh)
#         #if ( (valueOpenLow >= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= (pipdiff/SLTPRatio)) ):
#         #    trendcat[line] = 2 # bth limits exceeded
#         #elif ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
#         #    trendcat[line] = 3 #-1 downtrend
#         #elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
#         #    trendcat[line] = 1 # uptrend
#         #elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
#         #    trendcat[line] = 0 # no trend
#         #elif ( (valueOpenLow >= (pipdiff/SLTPRatio)) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
#         #    trendcat[line] = 5 # light trend down
#         #elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= (pipdiff/SLTPRatio)) ):
#         #    trendcat[line] = 4 # light trend up
#             if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
#                 trendcat[line] = 1 #-1 downtrend
#                 break
#             elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
#                 trendcat[line] = 2 # uptrend
#                 break
#             else:
#                 trendcat[line] = 0 # no clear trend
            
#     return trendcat
# #!!! pitfall one category High frequency
# df['Target'] = mytarget(30, df)
# #df.tail(20)
# #df['Target'] = df['Target'].astype(int)
# df['Target'].hist()
# import pandas_ta as pa
# df["RSI"] = pa.rsi(df.Close, length=16)
# df.tail(20)
# df.dropna(inplace=True)
# df.reset_index(drop=True,inplace=True)
# print(df.describe())
# attributes = ['RSI', 'signal', 'Target']
# df_model= df[attributes].copy()

# df_model['signal'] = pd.Categorical(df_model['signal'])
# dfDummies = pd.get_dummies(df_model['signal'], prefix = 'signalcategory')
# df_model= df_model.drop(['signal'], axis=1)
# df_model = pd.concat([df_model, dfDummies], axis=1)
# df_model
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, log_loss

# attributes = ['RSI', 'signalcategory_0', 'signalcategory_1', 'signalcategory_2']
# X = df_model[attributes]
# y = df_model['Target']

# train_pct_index = int(0.7 * len(X))
# X_train, X_test = X[:train_pct_index], X[train_pct_index:]
# y_train, y_test = y[:train_pct_index], y[train_pct_index:]

# model = XGBClassifier()
# model.fit(X_train, y_train)
# pred_train = model.predict(X_train)
# pred_test = model.predict(X_test)

# acc_train = accuracy_score(y_train, pred_train)
# acc_test = accuracy_score(y_test, pred_test)
# print('****Train Results****')
# print("Accuracy: {:.4%}".format(acc_train))
# print('****Test Results****')
# print("Accuracy: {:.4%}".format(acc_test))
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# matrix_train = confusion_matrix(y_train, pred_train)
# matrix_test = confusion_matrix(y_test, pred_test)

# print(matrix_train)
# print(matrix_test)

# report_train = classification_report(y_train, pred_train)
# report_test = classification_report(y_test, pred_test)

# print(report_train)
# print(report_test)
# #choices = [2, 0, -1, +1]
# ##choices = [2, 0, 3, +1]
# print(model.get_booster().feature_names)

# from matplotlib import pyplot
# from xgboost import plot_importance

# from sklearn.feature_selection import SelectFromModel
# #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
# #plot feature importance
# plot_importance(model)
# pyplot.show()
# print(model.get_booster().feature_names)
# from sklearn.neural_network import MLPClassifier

# attributes = ['RSI', 'signalcategory_0', 'signalcategory_1', 'signalcategory_2']
# X = df_model[attributes]
# y = df_model['Target']

# train_pct_index = int(0.6 * len(X))
# X_train, X_test = X[:train_pct_index], X[train_pct_index:]
# y_train, y_test = y[:train_pct_index], y[train_pct_index:]

# NN = MLPClassifier(hidden_layer_sizes=(50, 50, 60, 30, 9), random_state=100, verbose=0, max_iter=1000, activation='relu')
# NN.fit(X_train, y_train)
# pred_train = NN.predict(X_train)
# pred_test = NN.predict(X_test)
# acc_train = accuracy_score(y_train, pred_train)
# acc_test = accuracy_score(y_test, pred_test)
# print("="*20)

# print('****Train Results****')
# print("Accuracy: {:.4%}".format(acc_train))
# print('****Test Results****')
# print("Accuracy: {:.4%}".format(acc_test)) 
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# matrix_train = confusion_matrix(y_train, pred_train)
# matrix_test = confusion_matrix(y_test, pred_test)

# print(matrix_train)
# print(matrix_test)

# report_train = classification_report(y_train, pred_train)
# report_test = classification_report(y_test, pred_test)

# print(report_train)
# print(report_test)