import subprocess
import os
import time

# Directory where your scripts are located
scripts_directory = 'Crypto/coinglass_scripts'

# List all files in the scripts directory
scripts = os.listdir(scripts_directory)

# Filter out only .py files
py_scripts = [script for script in scripts if script.endswith('.py')]

# Run each Python script
for script in py_scripts:
    script_path = os.path.join(scripts_directory, script)
    print(f"Running {script}...")
    
    # Run the script, try to catch any errors
    try:
        time.sleep(20)
        subprocess.run(['python', script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e} trying to run again...")
        try:
            subprocess.run(['python', script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}, second fail")
            print('Waiting...')
            time.sleep(20)
            try:
                subprocess.run(['python', script_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}: {e}, third fail")
                continue


print("All scripts executed.")
