# main.py, runs all data prep scripts in order

import subprocess

def run_script(script_name):
    subprocess.run(['python', script_name])

def main():
    # List of scripts to run in order
    scripts = [
        'dataprepallmacro.py',
        'dataprepalltechnical.py',
        'fearandgreedall.py',
        'prepdata.py',
        'datacombiner.py'
    ]

    for script in scripts:
        run_script(script)
        print(f"{script} has been executed.")

if __name__ == "__main__":
    main()
