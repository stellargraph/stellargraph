# graphsage-test

Sandbox for testing graphsage implementations

## Running the example
Ensure you have a redis instance running locally on the default port 6379.
```
redis-server
```
Then run the example using:
```
python -m example.example_graphsage
```
Which writes the example PPI data to redis, then runs supervised graphsage in batches for 5 epochs.
