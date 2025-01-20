import requests
import zipfile
import os

# URL of the dataset
url = "https://universe.roboflow.com/ds/W62PcEudDX?key=MRrdVXzYxC"
zip_file_name = "roboflow.zip"

# Get the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
zip_file_path = os.path.join(script_dir, zip_file_name)

# Step 1: Download the dataset
print("Downloading dataset...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print("Download completed!")
else:
    print("Failed to download dataset")
    exit(1)

# Step 2: Unzip the dataset in the same directory as the script
print("Extracting dataset...")
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(script_dir)
print("Extraction completed!")

# Step 3: Remove the zip file
os.remove(zip_file_path)
print("Temporary zip file removed.")
