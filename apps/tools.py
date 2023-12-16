import os
from langchain.tools import tool
from dataclasses import dataclass,asdict
import requests
import json
from langchain.prompts import StringPromptTemplate
from langdetect import detect
@dataclass
class StockData:
    open: str
    high: str
    low: str
    close: str
    volume: str

@tool("stock or trade info", return_direct=False)
def get_stock(input:str,topk=5) ->str:
    """Useful for get one stock trade info; input must be the stock code"""
    
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
        print("JSON data is valid.")
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

@tool("translate", return_direct=False)
def translate_input(input_text):
    """Useful for translate input to English"""
    prompt = StringPromptTemplate(
        "Translate the following text to English:\n\nInput: {input}\n\nTranslate:"
    )
    return prompt.render(input=input_text)


def detect_language(text):
    text = remove_digits(text)
    print(text)
    lang = detect(text)
    return lang

def remove_digits(input_str):
    import re
    output_str = re.sub(r'\d+', '', input_str)
    return output_str

if __name__ == '__main__':
    print(detect_language("take a picture"))