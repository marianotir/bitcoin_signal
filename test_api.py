


import requests

url = 'http://localhost:5000/predict/bitcoin'
r = requests.get(url)

print(r.json())