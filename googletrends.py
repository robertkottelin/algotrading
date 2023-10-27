from pytrends.request import TrendReq
import pandas as pd

# Set up the trend fetching utility
pytrends = TrendReq(hl='en-US', tz=360)

# Build the payload for your request (add your keywords)
kw_list = ["S&P500"]  # You can add more keywords here if you like

pytrends.build_payload(kw_list, timeframe='all')

# Fetch the interest over time
interest_over_time_df = pytrends.interest_over_time()

# Since the data doesn't include the current date, we'll add it as the last date in the dataset.
if not interest_over_time_df.empty:
    latest_date = interest_over_time_df.index[-1]
    current_values = {keyword: [None] for keyword in kw_list}  # None or some default value
    current_df = pd.DataFrame(current_values, index=[latest_date])
    interest_over_time_df = interest_over_time_df.append(current_df)

# You might want to reset the index to get the date as a column
interest_over_time_df.reset_index(level=0, inplace=True)
interest_over_time_df.rename(columns={'index': 'Date'}, inplace=True)
interest_over_time_df.drop(columns=['isPartial'], inplace=True)


# Now 'Date' is a column and can be used for merging with other data frames on 'Date' column
print(interest_over_time_df)
