$base = "quant_news_trader"

$folders = @(
"$base",
"$base\src",
"$base\src\data",
"$base\src\features",
"$base\src\models",
"$base\src\utils",
"$base\notebooks"
)

foreach ($f in $folders) {
    New-Item -ItemType Directory -Force -Path $f | Out-Null
}

# requirements.txt
@"
pandas
numpy
yfinance
requests
PyYAML
scikit-learn
vaderSentiment
python-dotenv
"@ | Out-File "$base\requirements.txt" -Encoding utf8

# config.yaml
@"
market:
  ticker: "AAPL"
  lookback_days: 180
  interval: "1d"

news:
  provider: "newsapi"
  query: "Apple Inc"
  language: "en"
  page_size: 20

model:
  test_size: 0.2
  random_state: 42

logging:
  level: "INFO"
"@ | Out-File "$base\config.yaml" -Encoding utf8

# .env.example
@" 
NEWSAPI_KEY=YOUR_NEWSAPI_KEY_HERE
"@ | Out-File "$base\.env.example" -Encoding utf8

# main.py
@"
print("Pipeline ready. Start building your AI quant strategy here.")
"@ | Out-File "$base\src\main.py" -Encoding utf8
