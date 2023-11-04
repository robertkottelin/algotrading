import re

# Define the path to your log file
log_file_path = 'optuna_trials.log'

# Function to extract the trial value from a log line
def extract_trial_value(line):
    match = re.search(r'Trial finished with value: ([0-9.]+)', line)
    if match:
        return float(match.group(1))
    else:
        return None

# Read the log file and sort the lines
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# Sort lines by the trial value (descending order)
sorted_lines = sorted(lines, key=extract_trial_value, reverse=True)

# Write the sorted lines to a new log file or print them
with open('sorted_optuna_trials.log', 'w') as sorted_file:
    for line in sorted_lines:
        sorted_file.write(line)
