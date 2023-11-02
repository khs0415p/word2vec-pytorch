import urllib.request
import os

def load_data():
    if not os.path.exists('./data'):
        os.makedirs('./data', exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="./data/ratings.txt")