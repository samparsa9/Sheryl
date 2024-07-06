import pandas as pd
import requests
from bs4 import BeautifulSoup

def create_sp500_csv(output_path):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    df = df[['Symbol']].dropna()
    df.to_csv(output_path, index=False)