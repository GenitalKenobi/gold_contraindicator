import yfinance as yf
import pandas as pd
import tweepy
import datetime
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
# Generate this in the X Developer Portal
X_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAABpQ7wEAAAAAIZVA4YBfe3ozDO7ha6ksH0b9wjQ%3DGxXHpkbIIvRBx7qXpVAv9YXbKg9BphGEjZyBzUYwzf456zzciu" 

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_indian_gold_data(start_date, end_date):
    """
    Fetches daily stock data for Indian Gold ETF (Nippon India ETF Gold BeES).
    """
    ticker = "GOLDBEES.NS"
    gold_data = yf.download(ticker, start=start_date, end=end_date)
    
    gold_data.reset_index(inplace=True)
    gold_data['Date'] = gold_data['Date'].dt.date
    gold_data = gold_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return gold_data.set_index('Date')

def get_daily_tweets(query, target_date, bearer_token):
    """
    Fetches tweets for a specific day using the X API v2.
    """
    client = tweepy.Client(bearer_token=bearer_token)
    start_time = f"{target_date}T00:00:00Z"
    end_time = f"{target_date}T23:59:59Z"
    tweets_list = []
    
    try:
        response = client.search_recent_tweets(
            query=query,
            start_time=start_time,
            end_time=end_time,
            max_results=100, 
            tweet_fields=['created_at', 'text']
        )
        if response.data:
            for tweet in response.data:
                tweets_list.append(tweet.text)
                
    except tweepy.TweepyException as e:
        print(f"Error fetching tweets for {target_date}: {e}")
        
    return tweets_list

def analyze_daily_sentiment(tweets_list):
    """
    Calculates the average compound sentiment score for a list of tweets.
    Returns a score between -1.0 (highly negative) and 1.0 (highly positive).
    """
    if not tweets_list:
        return None
        
    daily_scores = []
    for tweet in tweets_list:
        # VADER polarity_scores returns a dictionary with pos, neu, neg, and compound scores
        score = analyzer.polarity_scores(tweet)
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
    gold_df = get_indian_gold_data(start_str, end_str)
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
                "Open": round(row['Open'], 2),
                "High": round(row['High'], 2),
                "Low": round(row['Low'], 2),
                "Close": round(row['Close'], 2),
                "Volume": row['Volume']
            }
        
        # 2. Get Tweets
        daily_tweets = get_daily_tweets("gold India -is:retweet", current_date, X_BEARER_TOKEN)
        
        # 3. Analyze Sentiment
        avg_sentiment = analyze_daily_sentiment(daily_tweets)
        
        # 4. Compile Record
        report_record = {
            "Date": current_date,
            **daily_financials,
            "Tweet_Count": len(daily_tweets),
            "Avg_Sentiment_Score": avg_sentiment,
            "Tweets": " | ".join(daily_tweets) 
        }
        reports.append(report_record)
        current_date += delta

    # Save to CSV
    new_df = pd.DataFrame(reports)
    filename = "gold_dataset.csv"
    
    if os.path.exists(filename):
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
    generate_daily_reports("2026-02-20", "2026-02-22")