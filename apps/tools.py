import os
from langchain.tools import tool
from dataclasses import dataclass,asdict
import requests
import json
from .model_factory import ModelFactory
from .prompt import stock_code_prompt

@dataclass
class StockData:
    open: str
    high: str
    low: str
    close: str
    volume: str

@tool("stock or trade info", return_direct=False)
def get_stock(input:str,topk=5) ->str:
    # """Useful for get one stock trade info; input must be the stock code"""
    """Useful for takeing the stock symbol or ticker as input and retrieves relevant trading data for that stock"""

    translate = ModelFactory.get_model("llama3")
    stock_code = stock_code_prompt(input)
    llm_out = translate.invoke(stock_code)
    input = parse_stock_code(llm_out.content)
    
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': input,
        'apikey': '1JXIUYN26HYID5Y9'
    }
    print(params)
    r = requests.get(url, params=params)
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("JSON data is invalid.")
        return "JSON data is invalid."

    if "Time Series (Daily)" not in data:
        return "JSON data does not contain 'Time Series (Daily)'."

    time_series_data = []
    for date, values in data["Time Series (Daily)"].items():
        stock_data = StockData(
            open=values["1. open"],
            high=values["2. high"],
            low=values["3. low"],
            close=values["4. close"],
            volume=values["5. volume"]
        )
        time_series_data.append((date, stock_data))

    serializable_data = {date: asdict(stock_data) for date, stock_data in time_series_data[:topk]}
    return json.dumps(serializable_data)

def get_stock_code(input:str):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'SYMBOL_SEARCH',
        'keywords': input,
        'apikey': '1JXIUYN26HYID5Y9'
    }
    print(params)
    r = requests.get(url, params=params)

def parse_stock_code(input:str):
    # Split the sentence into words
    words = input.split()
    # Loop through the words to find the stock symbol
    for word in words:
        if word.isupper():
            stock_symbol = word
            break
    return stock_symbol.rstrip('.')

# if __name__ == '__main__':
#     print(get_stock("MSFT"))