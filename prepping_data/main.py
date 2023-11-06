# main.py, runs all data prep scripts in order

import subprocess

def run_script(script_name):
    subprocess.run(['python', script_name])

def main():
    # List of scripts to run in order
    scripts = [
        'downloadSNP.py',
        'dataprepallmacro.py',
        'addmacro.py'
        'dataprepalltechnical.py',
        'fearandgreedall.py',
        'prepSNP.py'
    ]

    for script in scripts:
        run_script(script)
        print(f"{script} has been executed.")

if __name__ == "__main__":
    main()
