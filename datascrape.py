import yfinance as yf
import pandas as pd
from gnews import GNews
from newsapi import NewsApiClient
import datetime
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
# Initialize the GNews client
google_news = GNews(language='en', country='IN', max_results=10)

# Initialize NewsAPI using .env
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_stocks_data(ticker, start_date, end_date):
    """
    Fetches daily stock data for a given ticker from yfinance.
    """
    print(f"Downloading financials for {ticker}...")
    # Fetching with a small buffer to help with interpolation/backfill if range starts on weekend
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    buffer_start = (start_dt - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    
    data = yf.download(ticker, start=buffer_start, end=end_date)
    
    if data.empty:
        return pd.DataFrame()
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return data.set_index('Date')

def get_daily_gnews(query, target_date):
    """
    Fetches news articles for a specific day using GNews.
    """
    next_day = target_date + datetime.timedelta(days=1)
    google_news.start_date = (target_date.year, target_date.month, target_date.day)
    google_news.end_date = (next_day.year, next_day.month, next_day.day)
    
    news_list = []
    try:
        results = google_news.get_news(query)
        if results:
            for item in results:
                news_text = f"{item['title']}. {item.get('description', '')}"
                news_list.append(news_text)
    except Exception as e:
        print(f"Error fetching GNews for {query} on {target_date}: {e}")
        
    return news_list

def get_daily_newsapi(query, target_date):
    """
    Fetches news articles for a specific day using NewsAPI.org.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "your_actual_api_key_here":
        return []

    news_list = []
    try:
        date_str = target_date.strftime('%Y-%m-%d')
        response = newsapi.get_everything(
            q=query,
            from_param=date_str,
            to=date_str,
            language='en',
            sort_by='relevancy',
            page_size=10
        )
        
        if response['status'] == 'ok':
            for article in response['articles']:
                content = f"{article['title']}. {article.get('description', '')}"
                news_list.append(content)
                
    except Exception as e:
        print(f"Error fetching NewsAPI for {query} on {target_date}: {e}")
        
    return news_list

def analyze_sentiment(text_list):
    """
    Calculates the average compound sentiment score.
    Returns 0.0 if no text is provided.
    """
    if not text_list:
        return 0.0
    daily_scores = [analyzer.polarity_scores(text)['compound'] for text in text_list]
    return round(sum(daily_scores) / len(daily_scores), 4)

def get_news_query_for_ticker(ticker):
    """
    Maps a ticker symbol to a relevant news search query.
    """
    mapping = {
        "GOLDBEES.NS": "gold price India",
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "AAPL": "Apple stock",
        "BTC-USD": "Bitcoin",
        "GC=F": "Gold futures"
    }
    return mapping.get(ticker, f"{ticker} stock")

def generate_daily_reports(tickers, start_str, end_str):
    """
    Orchestrates data extraction with linear interpolation for stock data.
    """
    start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
    yf_end_date = (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        ticker_reports = []
        stock_df = get_stocks_data(ticker, start_str, yf_end_date)
        news_query = get_news_query_for_ticker(ticker)
        
        current_date = start_date
        while current_date <= end_date:
            print(f"[{ticker}] Processing {current_date}...")
            
            daily_financials = {"Open": None, "High": None, "Low": None, "Close": None, "Volume": None}
            if not stock_df.empty and current_date in stock_df.index:
                row = stock_df.loc[current_date]
                daily_financials = {
                    "Open": round(float(row['Open']), 2),
                    "High": round(float(row['High']), 2),
                    "Low": round(float(row['Low']), 2),
                    "Close": round(float(row['Close']), 2),
                    "Volume": int(row['Volume'])
                }
            
            gnews_articles = get_daily_gnews(news_query, current_date)
            gnews_sentiment = analyze_sentiment(gnews_articles)
            
            newsapi_articles = get_daily_newsapi(news_query, current_date)
            newsapi_sentiment = analyze_sentiment(newsapi_articles)
            
            report_record = {
                "Date": current_date,
                "Ticker": ticker,
                **daily_financials,
                "GNews_Count": len(gnews_articles),
                "GNews_Sentiment": gnews_sentiment,
                "NewsAPI_Count": len(newsapi_articles),
                "NewsAPI_Sentiment": newsapi_sentiment,
                "GNews_Titles": " | ".join(gnews_articles),
                "NewsAPI_Titles": " | ".join(newsapi_articles)
            }
            ticker_reports.append(report_record)
            current_date += datetime.timedelta(days=1)

        filename = f"{ticker}.csv"
        new_df = pd.DataFrame(ticker_reports)
        
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            existing_df = pd.read_csv(filename)
            existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.date
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        # Sort and deduplicate
        final_df = final_df.drop_duplicates(subset=['Date'], keep='last')
        final_df = final_df.sort_values(by='Date')

        # --- Data Interpolation ---
        stock_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Convert columns to numeric, forcing errors to NaN for interpolation
        for col in stock_cols:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        # 1. Linear interpolation for missing values in between
        final_df[stock_cols] = final_df[stock_cols].interpolate(method='linear')
        
        # 2. Backfill for the beginning (handles starting on weekends/holidays)
        final_df[stock_cols] = final_df[stock_cols].bfill()
        
        # 3. Forward fill for the very end if necessary
        final_df[stock_cols] = final_df[stock_cols].ffill()

        # Ensure volume stays as an integer type after interpolation
        final_df['Volume'] = final_df['Volume'].round().astype(int)

        final_df.to_csv(filename, index=False)
        print(f"Data for {ticker} saved and interpolated in {filename}")

if __name__ == "__main__":
    target_tickers = ["GOLDBEES.NS"]
    generate_daily_reports(target_tickers, "2026-02-01", "2026-02-27")
