# download_data.py
# This file downloads the bike sharing dataset into the data folder

import urllib.request
import zipfile

url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
zip_path = "data/bike_sharing.zip"

print("Downloading...")
urllib.request.urlretrieve(url, zip_path)

print("Unzipping...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("data")

print("Done! Check the data folder.")