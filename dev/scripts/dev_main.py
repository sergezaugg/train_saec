#--------------------------------
# Author : Serge Zaugg
# Description : dev script for interactive use 
#--------------------------------

import requests

repo = "sergezaugg/train_saec"
api_url = f"https://api.github.com/repos/{repo}/releases/latest"
# https://github.com/sergezaugg/train_saec/releases

response = requests.get(api_url)
data = response.json()

# Find the .whl asset URL
whl_url = None
for asset in data["assets"]:
    if asset["browser_download_url"].endswith(".whl"):
        whl_url = asset["browser_download_url"]
        break







