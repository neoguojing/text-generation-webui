import os

os.environ["ALPHAVANTAGE_API_KEY"] = '1JXIUYN26HYID5Y9'

def get_stock(input:str) ->str:
    """Useful for get one stock trade info; input must be the stock code"""
    import requests
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={input}&apikey={os.environ["ALPHAVANTAGE_API_KEY"]}'
    print(url)
    r = requests.get(url)
    data = r.json()['Time Series (Daily)']
    data = data[next(iter(data))]
    return data

if __name__ == '__main__':

    data = get_stock("MSFT")
    print(data)