import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import tweepy
import config
import finnhub
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import requests
import redis
import yfinance as yf
import yahoofinancials
from yahoofinancials import YahooFinancials
import cufflinks as cf
from cryptocmd import CmcScraper
import json
import pickle
import pandas_datareader as web
from datetime import datetime, date
from datetime import timedelta
from scipy.optimize import minimize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from psaw import PushshiftAPI
from yfinance import Ticker

st.set_page_config(layout="wide")
yf.pdr_override()
auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, config.TWITTER_CONSUME_SECRET)
auth.set_access_token(config.TWITTER_ACCESS_TOKEN, config.TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

with st.sidebar:
    Dashboard = option_menu("Dashboard", [
        "Overview", "Information", "Fundamental Analysis","News Test", "News & Analysis", 'StockTwits', 'Twitter', 'Reddit',
        'ML-Forecast', 'Portfolio Optimiser', 'ETF & Mutual Funds', "FOREX"], menu_icon="cast", default_index=0,
                            styles={
                                "nav-link"         : {"font-size"    : "16px", "text-align": "left", "margin": "0px",
                                                      "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "#2C3845"},
                            }
                            )

if Dashboard == 'Information':
    ticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/nitinnkdz/s-and-p-500-companies/master/data/constituents_symbols.txt',
        error_bad_lines=False)
    tickerSymbol = st.selectbox('Stock ticker', ticker_list)
    tickerData = yf.Ticker(tickerSymbol)  #
    tickerDf = tickerData.history(period="max")

    string_logo = '<img src=%s>' % tickerData.info['logo_url']
    st.markdown(string_logo, unsafe_allow_html=True)

    components.html(
        f"""
               <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div id="tradingview_769cb"></div>
          <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/symbols/{tickerSymbol}/" rel="noopener" target="_blank"><span class="blue-text">{tickerSymbol} Chart</span></a> by TradingView</div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
          "width": 980,
          "height": 610,
          "symbol": "{tickerSymbol}",
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "3",
          "locale": "in",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "show_popup_button": true,
          "popup_width": "1000",
          "popup_height": "650",
          "container_id": "tradingview_769cb"
        }}
          );
          </script>
        </div>
        <!-- TradingView Widget END -->
           """,
        height=610, width=980,
    )

    string_name = tickerData.info['longName']
    st.header('**%s**' % string_name)

    sec = tickerData.info['sector']
    st.write(sec)

    string_summary = tickerData.info['longBusinessSummary']
    st.info(string_summary)

    st.header('**Ticker data**')
    st.write(tickerDf)

    st.header('Major Holders')
    major_holders = tickerData.major_holders
    st._arrow_table(major_holders)

    st.header('Institutional Holders')
    institutional_holders = tickerData.institutional_holders
    st._arrow_table(institutional_holders)

    st.header('Mutual Funds Holders')
    mf_holders = tickerData.mutualfund_holders
    st._arrow_table(mf_holders)

    # Bollinger bands
    st.header('**Bollinger Bands**')
    qf = cf.QuantFig(tickerDf, title='First Quant Figure', legend='top', name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)
    
if Dashboard == 'News Test':
    def get_news(ticker):
	    url = finviz_url + ticker
	    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
	    response = urlopen(req)    
	    # Read the contents of the file into 'html'
	    html = BeautifulSoup(response)
	    # Find 'news-table' in the Soup and load it into 'news_table'
	    news_table = html.find(id='news-table')
	    return news_table
	
# parse news into dataframe
def parse_news(news_table):
    parsed_news = []
    
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text() 
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([date, time, text])        
        # Set column names
        columns = ['date', 'time', 'headline']
        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
        parsed_news_df = pd.DataFrame(parsed_news, columns=columns)        
        # Create a pandas datetime object from the strings in 'date' and 'time' column
        parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
        
    return parsed_news_df
        
def score_news(parsed_news_df):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    
    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')             
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')    
    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)          
    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

    return parsed_and_scored_news

def plot_hourly_sentiment(parsed_and_scored_news, ticker):
   
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('H').mean()

    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Hourly Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

def plot_daily_sentiment(parsed_and_scored_news, ticker):
   
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('D').mean()

    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Daily Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

# for extracting data from finviz
finviz_url = 'https://finviz.com/quote.ashx?t='

st.header("Bohmian's Stock News Sentiment Analyzer")

ticker = st.text_input('Enter Stock Ticker', '').upper()

try:
	st.subheader("Hourly and Daily Sentiment of {} Stock".format(ticker))
	news_table = get_news(ticker)
	parsed_news_df = parse_news(news_table)
	parsed_and_scored_news = score_news(parsed_news_df)
	fig_hourly = plot_hourly_sentiment(parsed_and_scored_news, ticker)
	fig_daily = plot_daily_sentiment(parsed_and_scored_news, ticker) 
	 
	st.plotly_chart(fig_hourly)
	st.plotly_chart(fig_daily)

	description = """
		The above chart averages the sentiment scores of {} stock hourly and daily.
		The table below gives each of the most recent headlines of the stock and the negative, neutral, positive and an aggregated sentiment score.
		The news headlines are obtained from the FinViz website.
		Sentiments are given by the nltk.sentiment.vader Python library.
		""".format(ticker)
		
	st.write(description)	 
	st.table(parsed_and_scored_news)
	
except:
	st.write("Enter a correct stock ticker, e.g. 'AAPL' above and hit Enter.")	

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

if Dashboard == 'News & Analysis':
    selected1 = option_menu(None, ["Market Crunch", "Ticker-News", "Analysis of News"],
                            menu_icon="cast", default_index=0, orientation="horizontal")

    if selected1 == 'Market Crunch':
        finnhub_client = finnhub.Client(api_key="c2tiabaad3i9opcku8r0")
        news = finnhub_client.general_news('general', min_id=0)
        for news in news:
            st.header(news['headline'])
            st.write(news['category'])
            posix_time = (news['datetime'])
            st.text(datetime.utcfromtimestamp(posix_time).strftime('%Y-%m-%dT%H:%M:%SZ'))
            st.image(news['image'])
            st.markdown(news['summary'])
            st.write(news['url'])

    if selected1 == 'Ticker-News':
        nsymbol = st.text_input("Enter the Ticker", value='TSLA', max_chars=10)
        url = f"https://api.polygon.io/v2/reference/news?limit=100&sort=published_utc&ticker={nsymbol}&published_utc.gte=2021-04-26&apiKey=l7CZdzU2ElYhYaDCj5QQeyVUxMgr7UPZ"
        r = requests.get(url)
        data = r.json()
        for results in data['results']:
            st.header(results['title'])
            st.write(results['author'])
            st.write(results['published_utc'])
            st.image(results['image_url'])
            st.write(results['article_url'])
            
    if selected1 == 'Analysis of News':
        def get_news(ticker):
            url = finviz_url + ticker
            req = Request(url=url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
            response = urlopen(req)
            # Read the contents of the file into 'html'
            html = BeautifulSoup(response, "html.parser")
            # Find 'news-table' in the Soup and load it into 'news_table'
            news_table = html.find(id='news-table')
            return news_table


        # parse news into dataframe
        def parse_news(news_table):
            global parsed_news_df
            parsed_news = []

            for x in news_table.findAll('tr'):
                # read the text from each tr tag into text
                # get text from a only
                text = x.a.get_text()
                # splite text in the td tag into a list
                date_scrape = x.td.text.split()
                # if the length of 'date_scrape' is 1, load 'time' as the only element

                if len(date_scrape) == 1:
                    time = date_scrape[0]

                # else load 'date' as the 1st element and 'time' as the second
                else:
                    date = date_scrape[0]
                    time = date_scrape[1]

                # Append ticker, date, time and headline as a list to the 'parsed_news' list
                parsed_news.append([date, time, text])
                # Set column names
                columns = ['date', 'time', 'headline']
                # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
                parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
                # Create a pandas datetime object from the strings in 'date' and 'time' column
                parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])

            return parsed_news_df


        def score_news(parsed_news_df):
            # Instantiate the sentiment intensity analyzer
            vader = SentimentIntensityAnalyzer()

            # Iterate through the headlines and get the polarity scores using vader
            scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()

            # Convert the 'scores' list of dicts into a DataFrame
            scores_df = pd.DataFrame(scores)

            # Join the DataFrames of the news and the list of dicts
            parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
            parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
            parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)
            parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

            return parsed_and_scored_news


        def plot_hourly_sentiment(parsed_and_scored_news, ticker):

            # Group by date and ticker columns from scored_news and calculate the mean
            mean_scores = parsed_and_scored_news.resample('H').mean()

            # Plot a bar chart with plotly
            fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score',
                         title=ticker + ' Hourly Sentiment Scores')
            return fig  # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later


        def plot_daily_sentiment(parsed_and_scored_news, ticker):

            # Group by date and ticker columns from scored_news and calculate the mean
            mean_scores = parsed_and_scored_news.resample('D').mean()

            # Plot a bar chart with plotly
            fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score',
                         title=ticker + ' Daily Sentiment Scores')
            return fig  # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later


        # for extracting data from finviz
        finviz_url = 'https://finviz.com/quote.ashx?t='

        st.header("Stock News Sentiment Analyzer")

        ticker = st.text_input('Enter Stock Ticker', '').upper()

        try:
            st.subheader("Hourly and Daily Sentiment of {} Stock".format(ticker))
            news_table = get_news(ticker)
            parsed_news_df = parse_news(news_table)
            parsed_and_scored_news = score_news(parsed_news_df)
            fig_hourly = plot_hourly_sentiment(parsed_and_scored_news, ticker)
            fig_daily = plot_daily_sentiment(parsed_and_scored_news, ticker)

            st.plotly_chart(fig_hourly)
            st.plotly_chart(fig_daily)

            description = """
        		The above chart averages the sentiment scores of {} stock hourly and daily.
        		The table below gives each of the most recent headlines of the stock and the negative, neutral, positive and an aggregated sentiment score.
        		The news headlines are obtained from the FinViz website.
        		Sentiments are given by the nltk.sentiment.vader Python library.
        		""".format(ticker)

            st.write(description)
            st.table(parsed_and_scored_news)

        except:
            st.write("Enter a correct stock ticker, e.g. 'AAPL' above and hit Enter.")

        hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if Dashboard == 'Twitter':
    for username in config.screenname:
        user = api.get_user(username)
        tweets = api.user_timeline(username)

        st.subheader(username)
        st.image(user.profile_image_url)

        for tweet in tweets:
            if '$' in tweet.text:
                words = tweet.text.split(' ')
                for word in words:
                    if word.startswith('$') and word[1:].isalpha():
                        symbol = word[1:]
                        st.write(symbol)
                        st.write(tweet.text)
                        st.image(f"https://finviz.com/chart.ashx?t={symbol}")

if Dashboard == 'Overview':
    st.markdown("This application allows you to examine Fundamentals, Market News, and Investor Sentiment.This application\
                can provide you with a variety of stock-related information, as well as forecast the future value of any\
                cryptocurrency/stocks! Under the hood, the application is developed with Streamlit (the front-end) and \
                the Facebook Prophet model, which is an advanced open-source forecasting model established by Facebook.\
                You can choose to train for any number of days in the future model on all available data or a specific \
                period range. Finally, the prediction results can be plotted on both a normal and log scale.This \
                application makes use of the Twitter API to deliver tweets connected to a ticker, as well as other APIs\
                such as Iex Cloud, Finnhub, and Polygon.io to provide various information\about a specific symbol. \
                This features trading-view to analyse charts as well.")

    components.html(
        """
        <!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com" rel="noopener" target="_blank"><span class="blue-text">Cryptocurrencies</span></a> <span class="blue-text">and</span> <a href="https://in.tradingview.com" rel="noopener" target="_blank"><span class="blue-text">Economy</span></a> by TradingView</div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-quotes.js" async>
  {
  "width": "800",
  "height": "500",
  "symbolsGroups": [
    {
      "name": "Indices",
      "originalName": "Indices",
      "symbols": [
        {
          "name": "FOREXCOM:SPXUSD",
          "displayName": "S&P 500"
        },
        {
          "name": "FOREXCOM:NSXUSD",
          "displayName": "US 100"
        },
        {
          "name": "FOREXCOM:DJI",
          "displayName": "Dow 30"
        },
        {
          "name": "INDEX:NKY",
          "displayName": "Nikkei 225"
        },
        {
          "name": "INDEX:DEU40",
          "displayName": "DAX Index"
        },
        {
          "name": "FOREXCOM:UKXGBP",
          "displayName": "UK 100"
        },
        {
          "name": "NSE:NIFTY",
          "displayName": "Nifty 50"
        },
        {
          "name": "INDEX:IBOV",
          "displayName": "IBovespa"
        },
        {
          "name": "INDEX:SMI",
          "displayName": "SMI"
        },
        {
          "name": "INDEX:CAC40",
          "displayName": "CAC 40"
        },
        {
          "name": "CURRENCYCOM:EU50",
          "displayName": "Eurostoxx 50"
        },
        {
          "name": "TVC:HSI",
          "displayName": "HSI"
        },
        {
          "name": "HOSE:VNINDEX",
          "displayName": "VNI Index"
        },
        {
          "name": "SET:SET",
          "displayName": "SET"
        },
        {
          "name": "OANDA:HK33HKD",
          "displayName": "Hong Kong 33"
        },
        {
          "name": "MOEX:IMOEX",
          "displayName": "Russia"
        },
        {
          "name": "OANDA:JP225USD",
          "displayName": "JP 225"
        },
        {
          "name": "KRX:KOSPI",
          "displayName": "KOSPI"
        },
        {
          "name": "BME:IBC",
          "displayName": "Spain"
        },
        {
          "name": "INDEX:SENSEX",
          "displayName": "BSE SENSEX"
        },
        {
          "name": "CRYPTOCAP:TOTAL",
          "displayName": "CRYPTO"
        },
        {
          "name": "NSE:BANKNIFTY",
          "displayName": "Banknifty"
        },
        {
          "name": "NSE:INDIAVIX",
          "displayName": "India Vix"
        }
      ]
    },
    {
      "name": "Futures",
      "originalName": "Futures",
      "symbols": [
        {
          "name": "TVC:GOLD",
          "displayName": "Gold"
        },
        {
          "name": "TVC:SILVER",
          "displayName": "Silver"
        },
        {
          "name": "CAPITALCOM:COPPER",
          "displayName": "Copper"
        },
        {
          "name": "TVC:USOIL",
          "displayName": "Crude Oil"
        },
        {
          "name": "TVC:UKOIL",
          "displayName": "Brent Oil"
        },
        {
          "name": "CURRENCYCOM:NATURALGAS",
          "displayName": "Natural Gas"
        },
        {
          "name": "NYMEX:RB1!",
          "displayName": "Gasoline RBOB"
        },
        {
          "name": "CBOT:ZW1!",
          "displayName": " US wheat"
        },
        {
          "name": "CBOT:ZC1!",
          "displayName": "US Corn"
        },
        {
          "name": "CBOT:ZS1!",
          "displayName": "US soyabean"
        },
        {
          "name": "ICEUS:KC1!",
          "displayName": "coffee"
        },
        {
          "name": "ICEUS:SB1!",
          "displayName": "Sugar 11"
        },
        {
          "name": "ICEUS:CT1!",
          "displayName": "Cotton #2"
        },
        {
          "name": "ICEEUR:C1!",
          "displayName": "London Cocoa"
        },
        {
          "name": "ICEUS:CC1!",
          "displayName": "US Cocoa"
        },
        {
          "name": "NYMEX:PL1!",
          "displayName": "Platinum"
        },
        {
          "name": "NYMEX:PA1!",
          "displayName": "Palladium"
        },
        {
          "name": "MOEX:AM1!",
          "displayName": "Aluminium"
        },
        {
          "name": "CME:LE1!",
          "displayName": "Live Cattle"
        },
        {
          "name": "CME:HET1!",
          "displayName": "Lean Hogs"
        },
        {
          "name": "COMEX:ZNC1!",
          "displayName": "Zinc"
        },
        {
          "name": "MOEX:NL1!",
          "displayName": "Nickel"
        },
        {
          "name": "COMEX:LED1!",
          "displayName": "Lead"
        }
      ]
    },
    {
      "name": "Bonds",
      "originalName": "Bonds",
      "symbols": [
        {
          "name": "CME:GE1!",
          "displayName": "Eurodollar"
        },
        {
          "name": "CBOT:ZB1!",
          "displayName": "T-Bond"
        },
        {
          "name": "CBOT:UB1!",
          "displayName": "Ultra T-Bond"
        },
        {
          "name": "EUREX:FGBL1!",
          "displayName": "Euro Bund"
        },
        {
          "name": "EUREX:FBTP1!",
          "displayName": "Euro BTP"
        },
        {
          "name": "EUREX:FGBM1!",
          "displayName": "Euro BOBL"
        },
        {
          "name": "TVC:US10Y",
          "displayName": "US-10Y"
        },
        {
          "name": "TVC:US30Y",
          "displayName": "US-30Y"
        },
        {
          "name": "TVC:US02Y",
          "displayName": "US-O2Y"
        },
        {
          "name": "TVC:JP10Y",
          "displayName": "JP-10Y"
        },
        {
          "name": "TVC:DE10Y",
          "displayName": "DE-10Y"
        },
        {
          "name": "TVC:GB10Y",
          "displayName": "GB-10Y"
        },
        {
          "name": "TVC:AU10Y",
          "displayName": "AU-10Y"
        },
        {
          "name": "TVC:CA10Y",
          "displayName": "CA-10Y"
        },
        {
          "name": "TVC:CN10Y",
          "displayName": "CN-10Y"
        },
        {
          "name": "TVC:IN10Y",
          "displayName": "IN-10Y"
        },
        {
          "name": "TVC:IT10Y",
          "displayName": "IT-1OY"
        },
        {
          "name": "TVC:NZ10Y",
          "displayName": "NZ-10Y"
        },
        {
          "name": "TVC:ZA10Y",
          "displayName": "ZA-10Y"
        },
        {
          "name": "TVC:BR10",
          "displayName": "BR-10Y"
        },
        {
          "name": "TVC:FR10",
          "displayName": "FR-10Y"
        },
        {
          "name": "TVC:ES10",
          "displayName": "ES-10Y"
        }
      ]
    },
    {
      "name": "Forex",
      "originalName": "Forex",
      "symbols": [
        {
          "name": "FX:EURUSD",
          "displayName": "EUR/USD"
        },
        {
          "name": "FX:GBPUSD",
          "displayName": "GBP/USD"
        },
        {
          "name": "FX:USDJPY",
          "displayName": "USD/JPY"
        },
        {
          "name": "FX:USDCHF",
          "displayName": "USD/CHF"
        },
        {
          "name": "FX:AUDUSD",
          "displayName": "AUD/USD"
        },
        {
          "name": "FX:USDCAD",
          "displayName": "USD/CAD"
        },
        {
          "name": "FX_IDC:USDINR",
          "displayName": "USD/INR"
        },
        {
          "name": "FX_IDC:GBPINR",
          "displayName": "GBP/INR"
        },
        {
          "name": "FX_IDC:EURINR",
          "displayName": "EUR/INR"
        },
        {
          "name": "FX:USDTRY",
          "displayName": "USD/TRY"
        },
        {
          "name": "FX:NZDUSD",
          "displayName": "NZD/USD"
        },
        {
          "name": "FX_IDC:USDRUB",
          "displayName": "USD/RUB"
        },
        {
          "name": "FX_IDC:USDCNY",
          "displayName": "USD/CNY"
        },
        {
          "name": "FX:EURNZD",
          "displayName": "EURNZD"
        },
        {
          "name": "FX_IDC:USDBRL",
          "displayName": "USD/BRL"
        },
        {
          "name": "FX:USDCNH",
          "displayName": "USD/CNH"
        }
      ]
    },
    {
      "name": "Cryptocurrencies",
      "symbols": [
        {
          "name": "BINANCE:BTCUSDT",
          "displayName": "Bitcoin"
        },
        {
          "name": "BINANCE:ETHUSDT",
          "displayName": "Ethereum"
        },
        {
          "name": "BINANCE:SOLUSDT",
          "displayName": "Sol"
        },
        {
          "name": "BINANCE:ADAUSDT",
          "displayName": "Cardano"
        },
        {
          "name": "BINANCE:GMTUSDT",
          "displayName": "GMT"
        },
        {
          "name": "BINANCE:AVAXUSDT",
          "displayName": "AVAX"
        },
        {
          "name": "BINANCE:BNBUSDT",
          "displayName": "Binance Coin"
        },
        {
          "name": "BINANCE:DOTUSDT",
          "displayName": "Dot"
        },
        {
          "name": "BINANCE:SHIBUSDT",
          "displayName": "Shib"
        },
        {
          "name": "BINANCE:GALAUSDT",
          "displayName": "Gala"
        },
        {
          "name": "BINANCE:WAVESUSDT",
          "displayName": "Waves"
        },
        {
          "name": "BINANCE:ATOMUSDT",
          "displayName": "Cosmos"
        }
      ]
    },
    {
      "name": "Economy",
      "symbols": [
        {
          "name": "ECONOMICS:USIRYY",
          "displayName": "US INFLATION RATE"
        },
        {
          "name": "FRED:UNRATE",
          "displayName": "UNEMPLOYMENT RATE"
        },
        {
          "name": "FRED:FEDFUNDS",
          "displayName": "EFFECTIVE FEDRAL FUNDS RATE"
        },
        {
          "name": "FRED:CPIAUCSL",
          "displayName": "CPI"
        },
        {
          "name": "FRED:BOGMBASE",
          "displayName": "MONETARY BASE"
        }
      ]
    }
  ],
  "showSymbolLogo": true,
  "colorTheme": "dark",
  "isTransparent": true,
  "locale": "in"
}
  </script>
</div>
<!-- TradingView Widget END -
   """,
        height=1100, width=1100,
    )

if Dashboard == 'StockTwits':

    symboltws = st.text_input("Symbol", value='AAPL', max_chars=10)
    r1 = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symboltws}.json")
    data1 = r1.json()
    for message in data1['messages']:
        st.image(message['user']['avatar_url'])
        st.write(message['user']['username'])
        st.write(message['created_at'])
        st.write(message['body'])

if Dashboard == 'ML-Forecast':
    selected3 = option_menu(None, ["Stocks", "Cryptocurrency", ],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    if selected3 == 'Stocks':
        START = st.date_input('Start Date', date(2011, 1, 1))
        END = st.date_input('End date')
        stockslist = pd.read_csv(
            'https://raw.githubusercontent.com/nitinnkdz/s-and-p-500-companies/master/data/constituents_symbols.txt',
            error_bad_lines=False)
        selected_stock = st.selectbox('Select the stock for prediction', stockslist)

        n_years = st.slider('Years of prediction:', 1, 5)
        period = n_years * 365


        @st.cache
        def load_data(ticker):
            data = yf.download(ticker, START, END)
            data.reset_index(inplace=True)
            return data


        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        st.subheader('Raw data')
        st.write(data.tail())


        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)


        plot_raw_data()

        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

    if selected3 == 'Cryptocurrency':
        st.markdown(
            """
        <style>
        .big-font {
            fontWeight: bold;
            font-size:22px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        ### Select ticker & number of days to predict on
        selected_ticker = st.text_input("Select a cryptocurrency", "BTC")
        period = int(
            st.number_input('Number of days to predict:', min_value=0, max_value=1000000, value=365, step=1))
        training_size = int(
            st.number_input('Training set (%) size:', min_value=10, max_value=100, value=100, step=5)) / 100


        ### Initialise scraper without time interval
        @st.cache
        def load_data(selected_ticker):
            init_scraper = CmcScraper(selected_ticker)
            df = init_scraper.get_dataframe()
            min_date = pd.to_datetime(min(df['Date']))
            max_date = pd.to_datetime(max(df['Date']))
            return min_date, max_date


        data_load_state = st.sidebar.text('Loading data...')
        min_date, max_date = load_data(selected_ticker)
        data_load_state.text('Loading data... done!')

        ### Select date range
        date_range = st.selectbox("Select the timeframe to train the model on:",
                                  options=["All available data", "Specific date range"])

        if date_range == "All available data":

            ### Initialise scraper without time interval
            scraper = CmcScraper(selected_ticker)

        elif date_range == "Specific date range":

            ### Initialise scraper with time interval
            start_date = st.date_input('Select start date:', min_value=min_date, max_value=max_date,
                                       value=min_date)
            end_date = st.date_input('Select end date:', min_value=min_date, max_value=max_date, value=max_date)
            scraper = CmcScraper(selected_ticker, str(start_date.strftime("%d-%m-%Y")),
                                 str(end_date.strftime("%d-%m-%Y")))

        ### Pandas dataFrame for the same data
        data = scraper.get_dataframe()

        st.subheader('Raw data')
        st.write(data.head())


        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)


        def plot_raw_data_log():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
            fig.update_yaxes(type="log")
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)


        ### Plot (log) data
        plot_log = st.checkbox("Plot log scale")
        if plot_log:
            plot_raw_data_log()
        else:
            plot_raw_data()

        if st.button("Predict"):

            df_train = data[['Date', 'Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

            ### Create Prophet model
            m = Prophet(
                changepoint_range=training_size,  # 0.8
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality=False,
                seasonality_mode='multiplicative',  # multiplicative/additive
                changepoint_prior_scale=0.05
            )

            ### Add (additive) regressor
            for col in df_train.columns:
                if col not in ["ds", "y"]:
                    m.add_regressor(col, mode="additive")

            m.fit(df_train)

            ### Predict using the model
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            ### Show and plot forecast
            st.subheader('Forecast data')
            st.write(forecast.head())

            st.subheader(f'Forecast plot for {period} days')
            fig1 = plot_plotly(m, forecast)
            if plot_log:
                fig1.update_yaxes(type="log")
            st.plotly_chart(fig1)

            st.subheader("Forecast components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)

if Dashboard == 'Portfolio Optimiser':
    symbol1list = pd.read_csv(
        'https://raw.githubusercontent.com/nitinnkdz/s-and-p-500-companies/master/data/constituents_symbols.txt',
        error_bad_lines=False)
    options = st.multiselect('Enter Tickers of the companies', symbol1list, ['SBIN.NS', 'ICICIBANK.NS'])

    # Setting start and end date with the datetime module
    start_date = st.date_input("Start Date", date(2015, 1, 1))
    end_date = st.date_input('End date')

    # Creating a empty Dataframe to store the data inside
    df = pd.DataFrame()
    end = datetime.today().strftime("%Y-%m-%d")
    for o in options:
        ticker = yf.Ticker(o)
        df[o] = ticker.history(start=start_date, end=end)["Close"]

    log_ret = np.log(df / df.shift(1))


    # Creating some helper functions to calculate the expected return, expected volatility and Sharpe Ratio

    def get_ret_vol_st(weights):
        weights = np.array(weights)
        ret = np.sum(log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        sr = ret / vol

        return np.array([ret, vol, sr])


    def neg_sharpe(weights):
        return get_ret_vol_st(weights)[2] * -1


    def check_sum(weights):
        return np.sum(weights) - 1


    # Creating some variables to store some values inside which will be needed for the minimize function from scipy

    cons = ({'type': 'eq', 'fun': check_sum})

    init_guess = [1 / len(options) for i in options]

    aaa = ((0, 1),)
    bounds = ()
    for i in range(0, len(options)):
        bounds = bounds + aaa

    # Calculating the optimal weights with the negative sharpe ratio
    opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

    # Plotting all the data that we created and showing the optimal weights and the Portfolio stats
    st.header("Closing Price of companys from Start Date - Today")
    st.line_chart(df)

    st.header("Data statistics")
    st.write(df.describe())

    st.header("Data Correlation")
    st.write(df.pct_change().corr())

    st.header("Optimal Portfolio Weights")
    for i in range(len(options)):
        st.write(options[i], opt_results.x[i])

    data = get_ret_vol_st(opt_results.x)

    st.header("Optimal Portfolio stats")
    st.write("Return: ", data[0])
    st.write("Volatility: ", data[1])
    st.write("Sharpe Ratio: ", data[2])

if Dashboard == 'Fundamental Analysis':
    selected2 = option_menu(None, ["Balance Sheet", "Income Statement", "Cash-Flow", 'Analyze Statements'],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    ticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/nitinnkdz/s-and-p-500-companies/master/data/constituents_symbols.txt',
        error_bad_lines=False)
    tickerSymbol = st.selectbox('Stock ticker', ticker_list)
    tickerData = yf.Ticker(tickerSymbol)
    if selected2 == 'Balance Sheet':
        tickerDf = tickerData.get_balance_sheet()
        st.subheader('**Balance Sheet**')
        st.table(tickerDf)

    if selected2 == 'Income Statement':
        tickerDfi = tickerData.financials
        st.subheader('**Income Statement**')
        st.table(tickerDfi)

    if selected2 == 'Cash-Flow':
        tickerDfc = tickerData.cashflow
        st.subheader('**Cash Flow**')
        st.table(tickerDfc)

    if selected2 == 'Analyze Statements':
        url23 = requests.get(
            f'https://financialmodelingprep.com/api/v3/ratios-ttm/{tickerSymbol}?apikey=7062ab81fcd13c8807496957c0208206')
        live_analysis = url23.json()
        st.markdown('P/E ratio')
        st.write(live_analysis[0]['peRatioTTM'])
        st.markdown('PE/G Ratio ')
        st.write(live_analysis[0]['pegRatioTTM'])
        st.markdown('Payout Ratio')
        st.write(live_analysis[0]['payoutRatioTTM'])
        st.markdown('Current ratio')
        st.write(live_analysis[0]['currentRatioTTM'])
        st.markdown('Quick ratio')
        st.write(live_analysis[0]['quickRatioTTM'])
        st.markdown('Cash ratio')
        st.write(live_analysis[0]['cashRatioTTM'])
        st.markdown('Days of sales Outstanding')
        st.write(live_analysis[0]['daysOfSalesOutstandingTTM'])
        st.markdown('Days of Inventory Outstanding')
        st.write(live_analysis[0]['daysOfInventoryOutstandingTTM'])
        st.markdown('Days of Payable Outstanding')
        st.write(live_analysis[0]['daysOfPayablesOutstandingTTM'])
        st.markdown('Operating Cycle')
        st.write(live_analysis[0]['operatingCycleTTM'])
        st.markdown('Cash Conversion Cycle')
        st.write(live_analysis[0]['cashConversionCycleTTM'])
        st.markdown('Gross Profit Margin')
        st.write(live_analysis[0]['grossProfitMarginTTM'])
        st.markdown('Operating Profit Margin')
        st.write(live_analysis[0]['operatingProfitMarginTTM'])
        st.markdown('Pretax Profit Margin')
        st.write(live_analysis[0]['pretaxProfitMarginTTM'])
        st.markdown('Net Profit Margin')
        st.write(live_analysis[0]['netProfitMarginTTM'])
        st.markdown('Effective Tax Rate')
        st.write(live_analysis[0]['effectiveTaxRateTTM'])
        st.markdown('Return on Assets')
        st.write(live_analysis[0]['returnOnAssetsTTM'])
        st.markdown('Return on Equity')
        st.write(live_analysis[0]['returnOnEquityTTM'])
        st.markdown('Return on Capital Employed')
        st.write(live_analysis[0]['returnOnCapitalEmployedTTM'])
        st.markdown('Net Income Per EBT')
        st.write(live_analysis[0]['netIncomePerEBTTTM'])
        st.markdown('EBT per EBIT')
        st.write(live_analysis[0]['ebtPerEbitTTM'])
        st.markdown('EBIT Per Revenue')
        st.write(live_analysis[0]['ebitPerRevenueTTM'])
        st.markdown('Debt Ratio')
        st.write(live_analysis[0]['debtRatioTTM'])
        st.markdown('Debt Equity Ratio')
        st.write(live_analysis[0]['debtEquityRatioTTM'])
        st.markdown('long Term Debt To Capitalization')
        st.write(live_analysis[0]['longTermDebtToCapitalizationTTM'])
        st.markdown('Total Debt To Capitalization')
        st.write(live_analysis[0]['totalDebtToCapitalizationTTM'])
        st.markdown('Interest Coverage')
        st.write(live_analysis[0]['interestCoverageTTM'])
        st.markdown('CashFlow To Debt Ratio')
        st.write(live_analysis[0]['cashFlowToDebtRatioTTM'])
        st.markdown('Company Equity Multiplier')
        st.write(live_analysis[0]['companyEquityMultiplierTTM'])
        st.markdown('ReceivablesTurnover')
        st.write(live_analysis[0]['receivablesTurnoverTTM'])
        st.markdown('Payables Turnover')
        st.write(live_analysis[0]['payablesTurnoverTTM'])
        st.markdown('Inventory Turnover')
        st.write(live_analysis[0]['inventoryTurnoverTTM'])
        st.markdown('FixedAsset Turnover')
        st.write(live_analysis[0]['fixedAssetTurnoverTTM'])
        st.markdown('Asset Turnover')
        st.write(live_analysis[0]['assetTurnoverTTM'])
        st.markdown('Operating Cash Flow Per Share')
        st.write(live_analysis[0]['operatingCashFlowPerShareTTM'])
        st.markdown('Free Cash Flow Per Share')
        st.write(live_analysis[0]['freeCashFlowPerShareTTM'])
        st.markdown('Cash Per Share')
        st.write(live_analysis[0]['cashPerShareTTM'])
        st.markdown('Operating Cash Flow Sales Ratio')
        st.write(live_analysis[0]['operatingCashFlowSalesRatioTTM'])
        st.markdown('Free Cash Flow Operating Cash Flow Ratio')
        st.write(live_analysis[0]['freeCashFlowOperatingCashFlowRatioTTM'])
        st.markdown('CashFlow Coverage Ratios')
        st.write(live_analysis[0]['cashFlowCoverageRatiosTTM'])
        st.markdown('Short Term Coverage Ratios')
        st.write(live_analysis[0]['shortTermCoverageRatiosTTM'])
        st.markdown('Capital Expenditure Coverage Ratio')
        st.write(live_analysis[0]['capitalExpenditureCoverageRatioTTM'])
        st.markdown('Dividend Paid And Capex Coverage Ratio')
        st.write(live_analysis[0]['dividendPaidAndCapexCoverageRatioTTM'])
        st.markdown('Price Book Value Ratio')
        st.write(live_analysis[0]['priceBookValueRatioTTM'])
        st.markdown('Price To Book Ratio')
        st.write(live_analysis[0]['priceToBookRatioTTM'])
        st.markdown('Price To Sales Ratio')
        st.write(live_analysis[0]['priceToSalesRatioTTM'])
        st.markdown('Price Earnings Ratio')
        st.write(live_analysis[0]['priceEarningsRatioTTM'])
        st.markdown('Price To Free Cash Flows Ratio')
        st.write(live_analysis[0]['priceToFreeCashFlowsRatioTTM'])
        st.markdown('Price To Operating Cash Flows Ratio')
        st.write(live_analysis[0]['priceToOperatingCashFlowsRatioTTM'])
        st.markdown('price Cash Flow Ratio')
        st.write(live_analysis[0]['priceCashFlowRatioTTM'])
        st.markdown('Rrice Earnings To Growth Ratio')
        st.write(live_analysis[0]['priceEarningsToGrowthRatioTTM'])
        st.markdown('Price Sales Ratio')
        st.write(live_analysis[0]['priceSalesRatioTTM'])
        st.markdown('Dividend Yield')
        st.write(live_analysis[0]['dividendYieldTTM'])
        st.markdown('Enterprise Value Multiple')
        st.write(live_analysis[0]['enterpriseValueMultipleTTM'])
        st.markdown('Price Fair Value')
        st.write(live_analysis[0]['priceFairValueTTM'])
        st.markdown('dividendPerShare')
        st.write(live_analysis[0]['dividendPerShareTTM'])

if Dashboard == 'Reddit':
    tsymbol = st.sidebar.text_input("Symbol", value='TSLA', max_chars=10)
    api = PushshiftAPI()
    start_time = date(2022, 5, 26)
    submissions = list(api.search_submissions(after=start_time,
                                              subreddit='wallstreetbets',
                                              filter=['url', 'author', 'title', 'subreddit'],
                                              limit=50)
                       )
    for submission in submissions:
        words = submission.title.split()
        cashtags = list(set(filter(lambda word: word.lower().startswith('$'), words)))
        if len(cashtags) > 0:
            st.write(cashtags)
            st.write(submission.title)

if Dashboard == 'ETF & Mutual Funds':
    selected2 = option_menu(None, ["ETF", "Mutual Funds", ],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    if selected2 == 'ETF':
        etf_list = pd.read_csv('https://raw.githubusercontent.com/nitinnkdz/Stocks-Dashboard/main/ETFs_Tickers.txt',
                               error_bad_lines=False)
        option = st.selectbox('Select the ETF', etf_list)
        etfData = yf.Ticker(option)

        etf_string_name = etfData.info['longName']
        st.subheader('**%s**' % etf_string_name)

        components.html(
            f"""
               <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div id="tradingview_769cb"></div>
          <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/symbols/{option}/" rel="noopener" target="_blank"><span class="blue-text">{option} Chart</span></a> by TradingView</div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
          "width": 980,
          "height": 610,
          "symbol": "{option}",
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "3",
          "locale": "in",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "show_popup_button": true,
          "popup_width": "1000",
          "popup_height": "650",
          "container_id": "tradingview_769cb"
        }}
          );
          </script>
        </div>
        <!-- TradingView Widget END -->
           """,
            height=610, width=980,
        )

        etf_string_summary = etfData.info['longBusinessSummary']
        st.info(etf_string_summary)

        st.caption('Holdings')
        components.html(
            f"""
                            <div style="border: 1px solid #e0e3eb; height: 610px; width: 100%">
                    <iframe width="100%" frameBorder="0"
                            height="100%"
                            src="https://widget.finnhub.io/widgets/etf-holdings?symbol={option}&theme=dark" title="{option} Holdings Data by Finnhub">
                    </iframe>
                 </div>
                 <div style="width: 100%; text-align: center; margin-top:10px;">
                        " target="_blank" style="color: #1db954;"></a>
                 </div>
                <!-- Finnhub Widget END -->
                   """,
            height=610, width=980
        )

        etf_beta3year = etfData.info['beta3Year']
        st.caption('beta3Year')
        st.write(etf_beta3year)

        st.caption('totalAssets')
        etf_total_assets = etfData.info['totalAssets']
        st.write(etf_total_assets)

        st.caption('fundFamily')
        etf_funds_family = etfData.info['fundFamily']
        st.write(etf_funds_family)

        st.caption('volume')
        etf_volume = etfData.info['volume']
        st.write(etf_volume)

        st.caption('Sector Weightings')
        etf_Sector_Weighting = etfData.info['sectorWeightings']
        st.dataframe(etf_Sector_Weighting)

        # etf_holdings = etfData.info['holdings']
        # st._legacy_table(etf_holdings)

        st.caption('Bond Ratings')
        etf_bonds_ratings = etfData.info['bondRatings']
        st.dataframe(etf_bonds_ratings)

        st.caption('Equity Holdings')
        etf_equity_holdings = etfData.info['equityHoldings']
        st.write(etf_equity_holdings)

    if selected2 == 'Mutual Funds':
        mf_ticker = pd.read_csv(
            'https://raw.githubusercontent.com/nitinnkdz/Stocks-Dashboard/main/Mutual_Funds_Tickers.txt',
            error_bad_lines=False)
        mf_option = st.selectbox('Select the MF', mf_ticker)
        mf_data = yf.Ticker(mf_option)

        mf_string_name = mf_data.info['longName']
        st.subheader('**%s**' % mf_string_name)

        mf_string_summary = mf_data.info['longBusinessSummary']
        st.info(mf_string_summary)

        st.caption('beta3Year')
        mf_beta3year = mf_data.info['beta3Year']
        st.write(mf_beta3year)

        st.caption('Moring Ster Risk Rating')
        mf_risk_rating = mf_data.info['morningStarRiskRating']
        st.write(mf_risk_rating)

        st.caption('YTD Return')
        mf_return = mf_data.info['ytdReturn']
        st.write(mf_return)

        st.caption('totalAssets')
        mf_total_assets = mf_data.info['totalAssets']
        st.write(mf_total_assets)

        st.caption('fundFamily')
        mf_funds_family = mf_data.info['fundFamily']
        st.write(mf_funds_family)

        st.caption('volume')
        mf_volume = mf_data.info['volume']
        st.write(mf_volume)

        st.caption('Sector Weightings')
        mf_Sector_Weighting = mf_data.info['sectorWeightings']
        st.dataframe(mf_Sector_Weighting)

        st.caption('Holdings')
        mf_holdings = mf_data.info['holdings']
        st._legacy_table(mf_holdings)

        st.caption('Bond Ratings')
        mf_bonds_ratings = mf_data.info['bondRatings']
        st.dataframe(mf_bonds_ratings)

        st.caption('Equity Holdings')
        mf_equity_holdings = mf_data.info['equityHoldings']
        st.write(mf_equity_holdings)

if Dashboard == 'FOREX':
    components.html(
        """
                    <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container">
              <div class="tradingview-widget-container__widget"></div>
              <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/markets/currencies/forex-heat-map/" rel="noopener" target="_blank"><span class="blue-text">Forex Heat Map</span></a> by TradingView</div>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-forex-heat-map.js" async>
              {
              "width": 1300,
              "height": 800,
              "currencies": [
                "EUR",
                "USD",
                "JPY",
                "GBP",
                "CHF",
                "AUD",
                "CAD",
                "NZD",
                "CNY",
                "TRY",
                "SEK",
                "NOK",
                "DKK",
                "ZAR",
                "HKD",
                "SGD",
                "THB",
                "MXN",
                "IDR",
                "KRW",
                "PLN",
                "ISK",
                "KWD",
                "PHP",
                "MYR",
                "INR",
                "TWD",
                "SAR",
                "AED",
                "RUB",
                "ILS",
                "ARS",
                "CLP",
                "COP",
                "PEN",
                "UYU"
              ],
              "isTransparent": false,
              "colorTheme": "light",
              "locale": "in"
            }
              </script>
            </div>
            <!-- TradingView Widget END -->
                    
         """,
        height=65000,
        width=1200,
    )

st.sidebar.write("Created By Nitin Kohli")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/nitin-kohli/)")
