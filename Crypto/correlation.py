import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Load the data
file_path = 'Crypto/coinglass_ta.csv'  # Replace with your file path
df = pd.read_csv(file_path)
df.drop(columns=['t'], inplace=True)  # Dropping the 't' column

# Compute correlation matrix
correlation_matrix = df.corr()

# Write the correlation matrix to the file
with open('Crypto/causality_correlation.txt', 'w') as f:
    f.write("Correlation Matrix:\n")
    f.write(correlation_matrix.to_string())
    f.write("\n\n")

# Granger Causality test requires time series data without NaN values
df = df.dropna()

# Define max lag for the test
max_lag = 12
causality_results = []

# Collect Granger causality test results for each column
for column in df.columns:
    if column != 'c':  # Skip the 'c' column itself
        test_result = grangercausalitytests(df[['c', column]], max_lag, verbose=False)
        p_values = [round(test_result[i+1][0]['ssr_chi2test'][1], 4) for i in range(max_lag)]
        min_p_value = min(p_values)
        causality_results.append({'Column': column, 'P-Values': p_values, 'Min P-Value': min_p_value})

# Sort the results by min p-value
sorted_results = sorted(causality_results, key=lambda x: x['Min P-Value'])

# Write sorted results to the file
with open('Crypto/causality_correlation_sorted.txt', 'w') as f:
    for result in sorted_results:
        f.write(f"Column: {result['Column']}\n")
        f.write(f"P-Values: {result['P-Values']}\n")
        f.write(f"Min P-Value: {result['Min P-Value']}\n\n")
