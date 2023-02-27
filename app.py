import json

import streamlit
import streamlit as st
import streamlit.components.v1 as components
import time
from datetime import datetime, date
from urllib.request import urlopen, Request
import cufflinks as cf
import finnhub
import numpy as np
import pandas as pd
import plotly.express as px
import quantstats as qs
import requests
import tweepy
import yfinance as yf
from PIL import Image
from bs4 import BeautifulSoup
from cryptocmd import CmcScraper
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
from psaw import PushshiftAPI
from scipy.optimize import minimize
from st_on_hover_tabs import on_hover_tabs
from streamlit_option_menu import option_menu
from openbb_terminal.sdk import openbb, TerminalStyle
from openbb_terminal.config_terminal import theme
import config

st.set_page_config(page_title="STOCKER", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://w0.peakpx.com/wallpaper/410/412/HD-wallpaper-plain-black-black.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         
         </style>
         """,
        unsafe_allow_html=True
    )


add_bg_from_url()

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
with st.sidebar:
    tabs = on_hover_tabs(
        tabName=["Home", "Market Overview", "Stocks", "News & Analysis", "Social Sentiments", "ETF & Mutual Funds",
                 "Startup Crunch", 'ML-Forecast', 'Portfolio', "FOREX-Map"], default_choice=0,
        iconName=['home', 'dashboard', 'candlestick_chart', 'feed', 'track_changes', 'receipt', 'economy',
                  'data_exploration', 'calculate', 'currency_rupee'],
        styles={'navtab': {'background-color': '#111',
                           'color': '#f5f0f0',
                           'font-size': '18px',
                           'transition': '.3s',
                           'white-space': 'nowrap',
                           'text-transform': 'uppercase'},
                'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                      'cursor': 'pointer'}},
                'iconStyle': {'position': 'fixed',
                              'left': '7.5px',
                              'text-align': 'left'},
                'tabStyle': {'list-style-type': 'none',
                             'margin-bottom': '30px',
                             'padding-left': '30px'}},
        key="1")

yf.pdr_override()
auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, config.TWITTER_CONSUME_SECRET)
auth.set_access_token(config.TWITTER_ACCESS_TOKEN, config.TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

if tabs == "Home":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        image = Image.open('AA.png')
        st.image(image, width=200)
    with col3:
        st.write(' ')
    st.markdown("This application allows you to examine Fundamentals, Market News, and Investor Sentiment.This application\
                can provide you with a variety of stock-related information, as well as forecast the future value of any\
                cryptocurrency/stocks! Under the hood, the application is developed with Streamlit (the front-end) and \
                the Facebook Prophet model, which is an advanced open-source forecasting model established by Facebook.\
                You can choose to train for any number of days in the future model on all available data or a specific \
                period range. Finally, the prediction results can be plotted on both a normal and log scale.This \
                application makes use of the Twitter API to deliver tweets connected to a ticker, as well as other APIs\
                such as Iex Cloud, Finnhub, and Polygon.io to provide various information \
                about a specific symbol.This features trading-view to analyse charts as well. ")

if tabs == 'Stocks':
    selected1 = option_menu(None, ["Ticker-Info", "Fundamentals"],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    if selected1 == 'Ticker-Info':
        tickerSymbol = st.text_input('Enter Tickers of the companies', 'AAPL')
        tickerData = yf.Ticker(tickerSymbol)
        tick = pd.DataFrame(tickerData.history(period="max"))

        col1, col2 = st.columns(2)
        with col1:
            url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={tickerSymbol}&apikey=8C6QMW4YWGS0X7E2'
            r = requests.get(url)
            dataaplfa = r.json()
            st.header(dataaplfa['Name'])
            st.write(dataaplfa['Description'])
            st.markdown('**Exchange**')
            st.write(dataaplfa['Exchange'])
            st.markdown('**Sector**')
            st.write(dataaplfa['Sector'])
            st.markdown('**Industry**')
            st.write(dataaplfa['Industry'])
        with col2:
            components.html(
                f"""
                    <!-- TradingView Widget BEGIN -->
                    <div class="tradingview-widget-container">
                      <div id="tradingview_c0b7d"></div>
                      <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/symbols/{tickerSymbol}/" rel="noopener" target="_blank"><span class="blue-text"></span></a></div>
                      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                      <script type="text/javascript">
                      new TradingView.MediumWidget(
                      {{
                      "symbols": [
                        [
                          "{tickerSymbol}",
                        ]
                      ],
                        "chartOnly": true,
                              "width": 800,
                              "height": 500,
                              "locale": "in",
                              "colorTheme": "dark",
                              "isTransparent": true,
                              "autosize": false,
                              "showVolume": false,
                              "hideDateRanges": false,
                              "scalePosition": "right",
                              "scaleMode": "Normal",
                              "fontFamily": "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
                              "noTimeScale": false,
                              "valuesTracking": "1",
                              "chartType": "area",
                              "fontColor": "#787b86",
                              "gridLineColor": "rgba(240, 243, 250, 0.06)",
                              "backgroundColor": "rgba(0, 0, 0, 1)",
                              "lineColor": "rgb(253,128,0)",
                              "topColor": "rgba(245, 124, 0, 0.3)",
                              "bottomColor": "rgba(41, 98, 255, 0)",
                              "lineWidth": 3,
                              "container_id": "tradingview_c0b7d"
                    }}
                      );
                      </script>
                    </div>
                """,
                height=610, width=980,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('**Ticker data**')
            st.write(tick)
        with col2:
            st.subheader('Major Holders')
            major_holders = tickerData.major_holders
            st._legacy_table(major_holders)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Mutual Funds Holders')
            mf_holders = tickerData.mutualfund_holders
            st._arrow_dataframe(mf_holders)
        with col2:
            st.subheader('Institutional Holders')
            institutional_holders = tickerData.institutional_holders
            st._arrow_dataframe(institutional_holders)

        #sec fillings
        st.subheader("SEC Fillings")
        finfill = openbb.stocks.fa.sec(symbol= tickerSymbol)
        st.write(finfill)

        col1, col2 = st.columns(2)
        with col1:
            # Government trades from openbb
            st.subheader("Congress Trading")
            st.pyplot(openbb.stocks.gov.gtrades_chart(symbol=tickerSymbol, gov_type='congress'))
        with col2:
            st.subheader("Insider Trading")
            insidert = openbb.stocks.ins.lins(symbol=tickerSymbol)
            st.write(insidert)

        # Technical Analysis Report from Finviz.brain
        st.subheader("Technical Analysis Summary")
        report = openbb.stocks.ta.summary(tickerSymbol)
        st.write(report)

        #contracts
        st.subheader('Contracts')
        contracts = openbb.stocks.gov.contracts(tickerSymbol)
        st.write(contracts)

        st.subheader(f"Lobbying effects by {tickerSymbol}")
        lobby = openbb.stocks.gov.lobbying(tickerSymbol)
        st.write(lobby)

        # Bollinger bands
        st.header('**Bollinger Bands**')
        qf = cf.QuantFig(tick, title='First Quant Figure', legend='top', name='GS')
        qf.add_bollinger_bands()
        fig = qf.iplot(asFigure=True)
        st.plotly_chart(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(openbb.ta.kc_chart(data = openbb.stocks.load(tickerSymbol)))
        with col2:
            st.pyplot(openbb.ta.obv_chart(data= openbb.stocks.load(tickerSymbol, start_date = '2022-11-18', interval = 5, prepost = True)))



    if selected1 == 'Fundamentals':
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

if tabs == 'News & Analysis':
    selected1 = option_menu(None, ["Market Crunch", "Ticker-News", "TestOpenbb"],
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

    if selected1 == 'TestOpenbb':
        def color_negative_red(val):
            if type(val) != 'str':
                color = 'green' if val > 0 else 'red'
                return f'color: {color}'


        st.title('openbb.economy')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('openbb.economy.currencies')
            data = openbb.economy.currencies()
            data[['Chng']] = data[['Chng']].apply(pd.to_numeric)
            st.dataframe(data.style.applymap(color_negative_red, subset=['Chng']))

        with col2:
            st.subheader('openbb.economy.usbonds')
            data = openbb.economy.usbonds()
            data[data.columns[1]] = data[data.columns[1]].apply(pd.to_numeric)
            data[data.columns[2]] = data[data.columns[2]].apply(pd.to_numeric)
            data[data.columns[3]] = data[data.columns[3]].apply(pd.to_numeric)

            columns = data.columns[3]

            st.dataframe(data.style.applymap(color_negative_red, subset=[columns]))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('openbb.economy.macro_countries')
            countries = pd.DataFrame.from_dict(openbb.economy.macro_countries(), orient='index')
            st.dataframe(countries)

        with col2:
            st.subheader('openbb.economy.indices')
            data = openbb.economy.indices()

            data[['Chg']] = data[['Chg']].apply(pd.to_numeric)

            st.dataframe(data.style.applymap(color_negative_red, subset=['Chg']))

        data = openbb.economy.treasury()
        st.subheader('openbb.economy.treasury')
        theme = TerminalStyle("dark", "dark", "dark")
        st.line_chart(data=data, x=None, y=None, width=0, height=0, use_container_width=True)

        st.subheader('openbb.economy.events')
        data = openbb.economy.events()
        st.dataframe(data)

if tabs == 'Market Overview':
    indicies, stocks, cryptoc, futures, forex = st.tabs(["Indicies", "stocks", "cryptoc", "futures", "forex"])
    with indicies:
        components.html(
            """
            <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container">
              <div class="tradingview-widget-container__widget"></div>
              <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/markets/indices/" rel="noopener" target="_blank"></a></div>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-quotes.js" async>
              {
              "title": "Indices",
              "width": 1250,
              "height": 750,
              "symbolsGroups": [
                {
                  "name": "US & Canada",
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
                      "name": "CME_MINI:ES1!",
                      "displayName": "S&P 500"
                    },
                    {
                      "name": "INDEX:DXY",
                      "displayName": "U.S. Dollar Currency Index"
                    },
                    {
                      "name": "FOREXCOM:DJI",
                      "displayName": "Dow 30"
                    }
                  ]
                },
                {
                  "name": "Europe",
                  "symbols": [
                    {
                      "name": "INDEX:SX5E",
                      "displayName": "Euro Stoxx 50"
                    },
                    {
                      "name": "FOREXCOM:UKXGBP",
                      "displayName": "UK 100"
                    },
                    {
                      "name": "INDEX:DEU40",
                      "displayName": "DAX Index"
                    },
                    {
                      "name": "INDEX:CAC40",
                      "displayName": "CAC 40 Index"
                    },
                    {
                      "name": "INDEX:SMI",
                      "displayName": "SWISS MARKET INDEX SMI® PRICE"
                    }
                  ]
                },
                {
                  "name": "Asia/Pacific",
                  "symbols": [
                    {
                      "name": "INDEX:NKY",
                      "displayName": "Nikkei 225"
                    },
                    {
                      "name": "INDEX:HSI",
                      "displayName": "Hang Seng"
                    },
                    {
                      "name": "BSE:SENSEX",
                      "displayName": "BSE SENSEX"
                    },
                    {
                      "name": "BSE:BSE500",
                      "displayName": "S&P BSE 500 INDEX"
                    },
                    {
                      "name": "INDEX:KSIC",
                      "displayName": "Kospi Composite"
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
            <!-- TradingView Widget END -->
            """,
            height=1250, width=2000,
        )
        with cryptoc:
            components.html(
                """
                 <!-- TradingView Widget BEGIN -->
                    <div class="tradingview-widget-container">
                      <div class="tradingview-widget-container__widget"></div>
                      <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/markets/cryptocurrencies/" rel="noopener" target="_blank"><span class="blue-text"></span></a> </div>
                      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-quotes.js" async>
                      {
                      "title": "Cryptocurrencies",
                      "title_raw": "Cryptocurrencies",
                      "title_link": "/markets/cryptocurrencies/prices-all/",
                      "width": 1250,
                      "height": 750,
                      "locale": "in",
                      "showSymbolLogo": true,
                      "symbolsGroups": [
                        {
                          "name": "Overview",
                          "symbols": [
                            {
                              "name": "CRYPTOCAP:TOTAL"
                            },
                            {
                              "name": "BITSTAMP:BTCUSD"
                            },
                            {
                              "name": "BITSTAMP:ETHUSD"
                            },
                            {
                              "name": "FTX:SOLUSD"
                            },
                            {
                              "name": "BINANCE:AVAXUSD"
                            },
                            {
                              "name": "COINBASE:UNIUSD"
                            }
                          ]
                        },
                        {
                          "name": "Bitcoin",
                          "symbols": [
                            {
                              "name": "BITSTAMP:BTCUSD"
                            },
                            {
                              "name": "COINBASE:BTCEUR"
                            },
                            {
                              "name": "COINBASE:BTCGBP"
                            },
                            {
                              "name": "BITFLYER:BTCJPY"
                            },
                            {
                              "name": "CEXIO:BTCRUB"
                            },
                            {
                              "name": "CME:BTC1!"
                            }
                          ]
                        },
                        {
                          "name": "Ethereum",
                          "symbols": [
                            {
                              "name": "BITSTAMP:ETHUSD"
                            },
                            {
                              "name": "KRAKEN:ETHEUR"
                            },
                            {
                              "name": "COINBASE:ETHGBP"
                            },
                            {
                              "name": "BITFLYER:ETHJPY"
                            },
                            {
                              "name": "BINANCE:ETHBTC"
                            },
                            {
                              "name": "BINANCE:ETHUSDT"
                            }
                          ]
                        },
                        {
                          "name": "Solana",
                          "symbols": [
                            {
                              "name": "FTX:SOLUSD"
                            },
                            {
                              "name": "BINANCE:SOLEUR"
                            },
                            {
                              "name": "COINBASE:SOLGBP"
                            },
                            {
                              "name": "BINANCE:SOLBTC"
                            },
                            {
                              "name": "HUOBI:SOLETH"
                            },
                            {
                              "name": "BINANCE:SOLUSDT"
                            }
                          ]
                        },
                        {
                          "name": "Uniswap",
                          "symbols": [
                            {
                              "name": "COINBASE:UNIUSD"
                            },
                            {
                              "name": "KRAKEN:UNIEUR"
                            },
                            {
                              "name": "COINBASE:UNIGBP"
                            },
                            {
                              "name": "BINANCE:UNIBTC"
                            },
                            {
                              "name": "KRAKEN:UNIETH"
                            },
                            {
                              "name": "BINANCE:UNIUSDT"
                            }
                          ]
                        }
                      ],
                       "colorTheme": "dark",
                       "isTransparent": true
                    }
                      </script>
                    </div>
                    <!-- TradingView Widget END -->
                """,
                height=1250, width=2000,
            )

if tabs == 'Social Sentiments':
    selected1 = option_menu(None, ["StockTwits", "Twitter", "Reddit"],
                            menu_icon="cast", default_index=0, orientation="horizontal")

    if selected1 == 'StockTwits':
        symboltws = st.text_input("Symbol", value='AAPL', max_chars=10)
        r1 = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symboltws}.json")
        data1 = r1.json()
        for message in data1['messages']:
            st.image(message['user']['avatar_url'])
            st.write(message['user']['username'])
            st.write(message['created_at'])
            st.write(message['body'])

    if selected1 == 'Twitter':
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

    if selected1 == 'Reddit':
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

if tabs == 'ML-Forecast':
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

if tabs == 'Portfolio':
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

if tabs == 'ETF & Mutual Funds':
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
                              <div id="tradingview_c0b7d"></div>
                              <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/symbols/{option}/" rel="noopener" target="_blank"><span class="blue-text"></span></a></div>
                              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                              <script type="text/javascript">
                              new TradingView.MediumWidget(
                              {{
                              "symbols": [
                                [
                                  "{option}",
                                ]
                              ],
                              "chartOnly": true,
                              "width": 1000,
                              "height": 500,
                              "locale": "in",
                              "colorTheme": "dark",
                              "isTransparent": true,
                              "autosize": false,
                              "showVolume": false,
                              "hideDateRanges": false,
                              "scalePosition": "right",
                              "scaleMode": "Normal",
                              "fontFamily": "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
                              "noTimeScale": false,
                              "valuesTracking": "1",
                              "chartType": "area",
                              "fontColor": "#787b86",
                              "gridLineColor": "rgba(240, 243, 250, 0.06)",
                              "lineColor": "",
                              "topColor": "rgba(41, 98, 255, 0.3)",
                              "bottomColor": "rgba(41, 98, 255, 0)",
                              "lineWidth": 3,
                              "container_id": "tradingview_c0b7d"
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

if tabs == 'FOREX-Map':
    with st.spinner('Wait for it...'):
        time.sleep(5)
        st.success('Done!')
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

if tabs == 'Quant Report':
    qs.extend_pandas()
    qticker_list = pd.read_csv(
        'https://raw.githubusercontent.com/nitinnkdz/s-and-p-500-companies/master/data/constituents_symbols.txt',
        error_bad_lines=False)
    QtSymbol = st.selectbox('Stock ticker', qticker_list)
    stock_returns = qs.utils.download_returns(QtSymbol)
    sharpe = qs.stats.sharpe(stock_returns)
    st.write(sharpe)
    qs.reports.html(stock_returns, QtSymbol)
    HtmlFile = open("tearsheet.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    print(source_code)
    components.html(source_code)

if tabs == 'Startup Crunch':
    jig = st.text_input("Symbol", value='Startup', max_chars=15)
    url = ('https://newsapi.org/v2/everything?'
           f'q={jig}&'
           'domains= indiatimes.com,vccircle.com,techcrunch.com,moneycontrol.com,business-standard.com,livemint.com&'
           'apiKey=e9281231a2bb483caaeccce82d9a235d')

    response = requests.get(url)
    articles = json.loads(response.text)['articles']
    for articles in articles:
        st.header(articles["title"])
        st.write(articles["publishedAt"], articles["author"])
        st.markdown(articles["description"])
        st.image(articles["urlToImage"])
        with st.expander('Expand'):
            st.write(articles["content"])
