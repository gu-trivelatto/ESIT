import subprocess
import os

# Change to the desired directory
os.chdir('../CESM')

# Define the command to run the script
command = ['python', 'cesm.py', 'run', '-m', 'DEasModel', '-s', 'Base']

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output
print(result.stdout)
if result.stderr:
    print(result.stderr)