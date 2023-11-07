# main.py, runs all data prep scripts in order

import subprocess

def run_script(script_name):
    subprocess.run(['python', script_name])

def main():
    # List of scripts to run in order
    scripts = [
        'prepping_data/downloadSNP.py',
        'prepping_data/dataprepallmacro.py',
        'prepping_data/addmacro.py',
        'prepping_data/dataprepalltechnical.py',
        'prepping_data/fearandgreedall.py',
        'prepping_data/prepSNP.py'
    ]

    for script in scripts:
        run_script(script)
        print(f"{script} has been executed.")

if __name__ == "__main__":
    main()
