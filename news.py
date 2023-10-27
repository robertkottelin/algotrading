from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

load_dotenv()
# Get the API key from environment variables
news_key = os.getenv('NEWS_KEY')

# Init
newsapi = NewsApiClient(api_key=news_key)

# # /v2/top-headlines
# top_headlines = newsapi.get_top_headlines(q='bitcoin',
#                                           sources='bbc-news,the-verge',
#                                           category='business',
#                                           language='en',
#                                           country='us')

# /v2/everything
all_articles = newsapi.get_everything(q='bitcoin',
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2023-10-01',
                                      to='2023-10-27',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)

# /v2/top-headlines/sources
sources = newsapi.get_sources()

print(sources)