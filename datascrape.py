import yfinance as yf
import pandas as pd
from gnews import GNews
import datetime
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
# Initialize the GNews client
google_news = GNews(language='en', country='IN', max_results=20)

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_indian_gold_data(start_date, end_date):
    """
    Fetches daily stock data for Indian Gold ETF (Nippon India ETF Gold BeES).
    """
    ticker = "GOLDBEES.NS"
    gold_data = yf.download(ticker, start=start_date, end=end_date)
    
    if gold_data.empty:
        return pd.DataFrame()
    
    # Flatten MultiIndex columns if necessary
    if isinstance(gold_data.columns, pd.MultiIndex):
        gold_data.columns = gold_data.columns.get_level_values(0)
        
    gold_data.reset_index(inplace=True)
    gold_data['Date'] = gold_data['Date'].dt.date
    gold_data = gold_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return gold_data.set_index('Date')

def get_daily_news(query, target_date):
    """
    Fetches news articles for a specific day using GNews.
    """
    # Set the date range for the target day
    # GNews expects start_date and end_date as (year, month, day)
    next_day = target_date + datetime.timedelta(days=1)
    
    google_news.start_date = (target_date.year, target_date.month, target_date.day)
    google_news.end_date = (next_day.year, next_day.month, next_day.day)
    
    news_list = []
    
    try:
        # Fetch news
        results = google_news.get_news(query)
        if results:
            for item in results:
                # We'll use the title and description for sentiment analysis
                news_text = f"{item['title']}. {item.get('description', '')}"
                news_list.append(news_text)
                
    except Exception as e:
        print(f"Error fetching news for {target_date}: {e}")
        
    return news_list

def analyze_daily_sentiment(text_list):
    """
    Calculates the average compound sentiment score for a list of text strings.
    Returns a score between -1.0 (highly negative) and 1.0 (highly positive).
    """
    if not text_list:
        return None
        
    daily_scores = []
    for text in text_list:
        # VADER polarity_scores returns a dictionary with pos, neu, neg, and compound scores
        score = analyzer.polarity_scores(text)
        daily_scores.append(score['compound'])
        
    # Return the average compound score for the day
    return round(sum(daily_scores) / len(daily_scores), 4)

def generate_daily_reports(start_str, end_str):
    """
    Orchestrates the data extraction, sentiment analysis, and reporting.
    """
    start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

    print(f"Fetching Gold data from {start_date} to {end_date}...")
    gold_df = get_indian_gold_data(start_str, (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    reports = []

    delta = datetime.timedelta(days=1)
    current_date = start_date
    
    while current_date <= end_date:
        print(f"Processing data for {current_date}...")
        
        # 1. Get Financials
        daily_financials = {"Open": None, "High": None, "Low": None, "Close": None, "Volume": None}
        if current_date in gold_df.index:
            row = gold_df.loc[current_date]
            daily_financials = {
                "Open": round(float(row['Open']), 2),
                "High": round(float(row['High']), 2),
                "Low": round(float(row['Low']), 2),
                "Close": round(float(row['Close']), 2),
                "Volume": int(row['Volume'])
            }
        
        # 2. Get News
        daily_news = get_daily_news("gold price India", current_date)
        
        # 3. Analyze Sentiment
        avg_sentiment = analyze_daily_sentiment(daily_news)
        
        # 4. Compile Record
        report_record = {
            "Date": current_date,
            **daily_financials,
            "News_Count": len(daily_news),
            "Avg_Sentiment_Score": avg_sentiment,
            "News_Titles": " | ".join(daily_news) 
        }
        reports.append(report_record)
        current_date += delta

    # Save to CSV
    new_df = pd.DataFrame(reports)
    filename = "gold_dataset.csv"
    
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        existing_df = pd.read_csv(filename)
        # Convert Date column to date objects for consistent comparison
        existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.date
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    # Remove duplicates (keep latest) and sort chronologically
    final_df = final_df.drop_duplicates(subset=['Date'], keep='last')
    final_df = final_df.sort_values(by='Date')
    
    final_df.to_csv(filename, index=False)
    print(f"Dataset updated and saved to {filename}")

# --- Execution ---
if __name__ == "__main__":
    # Example range for testing (already available in GNews)
    generate_daily_reports("2026-02-20", "2026-02-22")