import yfinance as yf
import pandas as pd
from gnews import GNews
import datetime
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
# Initialize the GNews client
google_news = GNews(language='en', country='IN', max_results=10)

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Examples of tickers you can use:
# "GOLDBEES.NS" - Nippon India ETF Gold BeES (India)
# "RELIANCE.NS" - Reliance Industries Limited (India)
# "TCS.NS"      - Tata Consultancy Services Limited (India)
# "AAPL"        - Apple Inc. (USA)
# "BTC-USD"     - Bitcoin USD (Crypto)
# "GC=F"        - Gold Futures (Global)

def get_stocks_data(ticker, start_date, end_date):
    """
    Fetches daily stock data for a given ticker from yfinance.
    """
    print(f"Downloading financials for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        return pd.DataFrame()
    
    # Flatten MultiIndex columns if necessary (common in newer yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return data.set_index('Date')

def get_daily_news(query, target_date):
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
        print(f"Error fetching news for {query} on {target_date}: {e}")
        
    return news_list

def analyze_daily_sentiment(text_list):
    """
    Calculates the average compound sentiment score for a list of text strings.
    """
    if not text_list:
        return None
        
    daily_scores = []
    for text in text_list:
        score = analyzer.polarity_scores(text)
        daily_scores.append(score['compound'])
        
    return round(sum(daily_scores) / len(daily_scores), 4)

def get_news_query_for_ticker(ticker):
    """
    Maps a ticker symbol to a relevant news search query.
    """
    mapping = {
        "GOLDBEES.NS": "gold price India",
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "AAPL": "Apple stock news",
        "BTC-USD": "Bitcoin price",
        "GC=F": "Gold futures market"
    }
    # Default to the ticker itself if not in mapping
    return mapping.get(ticker, f"{ticker} stock")

def generate_daily_reports(tickers, start_str, end_str):
    """
    Orchestrates data extraction for multiple tickers.
    """
    start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

    all_reports = []
    
    # We fetch a bit more financial data to cover the date range properly
    yf_end_date = (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        stock_df = get_stocks_data(ticker, start_str, yf_end_date)
        news_query = get_news_query_for_ticker(ticker)
        
        current_date = start_date
        while current_date <= end_date:
            print(f"[{ticker}] Processing {current_date}...")
            
            # 1. Get Financials
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
            
            # 2. Get News (Synchronized with ticker)
            daily_news = get_daily_news(news_query, current_date)
            
            # 3. Analyze Sentiment
            avg_sentiment = analyze_daily_sentiment(daily_news)
            
            # 4. Compile Record
            report_record = {
                "Date": current_date,
                "Ticker": ticker,
                **daily_financials,
                "News_Count": len(daily_news),
                "Avg_Sentiment_Score": avg_sentiment,
                "News_Titles": " | ".join(daily_news) 
            }
            all_reports.append(report_record)
            current_date += datetime.timedelta(days=1)

    # Save to CSV
    new_df = pd.DataFrame(all_reports)
    filename = "gold_dataset.csv" # Keeping filename but note it contains more than gold now
    
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        existing_df = pd.read_csv(filename)
        existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.date
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    # Remove duplicates based on Date AND Ticker
    final_df = final_df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    final_df = final_df.sort_values(by=['Date', 'Ticker'])
    
    final_df.to_csv(filename, index=False)
    print(f"\nDataset updated and saved to {filename}")

# --- Execution ---
if __name__ == "__main__":
    # You can now pass a list of tickers
    target_tickers = ["GOLDBEES.NS", "RELIANCE.NS"]
    generate_daily_reports(target_tickers, "2026-02-20", "2026-02-22")
