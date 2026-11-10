from download import load_secret
from massive import RESTClient

#############################################################
#client = RESTClient(api_key=load_secret(), pagination=False)

#quotes = []
#for t in client.list_quotes(
#	ticker="SKHY",
#	order="asc",
#	limit=10,
#	sort="timestamp",
#	):
#    quotes.append(t)

#print(quotes)
#############################################################

import duckdb

connection = duckdb.connect(
    "../data/skhynix.duckdb",
    read_only=True,
)

events = connection.execute("""
    SELECT *
    FROM event_korea_asof
    WHERE anchor_type = 'premarket_0400'
    ORDER BY event_date_ny
""").df()

print(events.head())