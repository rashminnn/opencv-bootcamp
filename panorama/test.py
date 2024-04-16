import glob
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the current working directory
cwd = os.getcwd()
print("Current working directory:", cwd)


# Use glob to find all JPEG files in the "boat" directory
imagefiles = glob.glob(f"{cwd}{os.sep}*.jpg")
imagefiles.sort()

# Print the list of found files
print("Found files:", imagefiles)

# Continue with your image loading and display code..


# List all files in the "boat" directory
boat_files = os.listdir(imagefiles)
print("Files in the 'boat' directory:", boat_files)