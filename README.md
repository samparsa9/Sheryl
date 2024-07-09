# Files and Modules
## crypto_scraper.py
  Python script to create a starter CSV file containing symbols of selected cryptocurrencies for clustering.
## sp500_scraper.py
  Python script to scrape Wikipedia for a list of current S&P 500 companies and create a CSV file with their symbols.
## k_means_cluster.py
  Python script defining functions for applying K-Means clustering to financial data.
## scaler.py
  Python script defining functions for scaling data using StandardScaler from scikit-learn.
## plotting.py
  ### Python script containing functions for visualizing data, including:
    plot_features: Plots outlier detection using KNN, Isolation Forest, and LOF algorithms on selected feature pairs.
    plot_features_1: Plots correlation heatmap, pair plot, and descriptive statistics of selected features.
    plot_features_2: Plots outlier detection using KNN, Isolation Forest, and LOF algorithms on all features.
    plot_features_3: Plots correlation matrix, 3D scatter plot, parallel coordinates plot, and heatmap of mean feature values by cluster.
    plot_features_4: Plots outlier detection using KNN and Isolation Forest algorithms on selected features.
    plot_features_interactive: Plots interactive PCA scatter plot, correlation heatmap, 3D scatter plot, parallel coordinates plot, and heatmap of mean feature values by cluster.
    plot_features_interactive_1: Plots interactive PCA scatter plot with animation, sunburst chart for cluster composition, radar chart for feature comparison across clusters, interactive heatmap with dendrogram, and interactive violin plot for feature distribution.
## JOKR_strat.py
  Python script for a machine learning-based trading algorithm using K-Means clustering and portfolio optimization. Handles data collection, preprocessing, clustering, and automated trading decisions based on current market conditions.
## alpaca_utils.py
  Python script containing utility functions for interacting with the Alpaca API, including retrieving account information, executing trades, and setting up portfolio clusters for trading strategies.
## email_utils.py
  Python script for sending emails using SMTP. Utilizes environment variables for sender, recipient, and password setup.
## misc.py
  ### Python script with miscellaneous functions:
    Is_balanced: Checks if current portfolio is balanced within a specified threshold.
    calculate_seconds_till_next_reallocation: Calculates seconds until the next reallocation time based on specified trade hour and minute.
## main.py
  ### Python script to orchestrate the entire pipeline (seperate from trading algorithm, only sets up
  ### the current vs optimal portfolio dataframe and plots relevant data related to K-means algorithm):
    Imports necessary modules and configurations.
    Executes data scraping (cryptocurrency or S&P 500), feature calculation, data scaling, K-Means clustering, and portfolio setup.
    Utilizes interactive plotting for visualization.
    Includes a main function for execution based on crypto flag.
## index.html
HTML file for displaying portfolio overview and positions using Chart.js and Bootstrap.
## app.py
  ### Flask application to serve index.html and fetch portfolio data from AWS DynamoDB.
    Initializes DynamoDB resource and defines tables for portfolio overview and positions.
    Includes routes /portfolio-overview and /portfolio-positions/<overview_id> to retrieve data.
## dynamo.py
  ### Python script for managing interactions with AWS DynamoDB:
    Initializes DynamoDB resource and defines tables for portfolio overview and positions.
    Includes functions to upload portfolio metrics including total account value and current positions.
## requirements.txt
### Text file listing dependencies for the project:
  ```
  alpaca-trade-api
  numpy
  pandas
  yfinance
  requests
  beautifulsoup4
  scikit-learn
  matplotlib
  python-dotenv
  pytz
  fredapi
  sqlalchemy
  schedule
  django
  ```
# Usage
## Setup:
  Clone the repository and install required dependencies listed in requirements.txt.
## Data Scraping:
  Use crypto_scraper.py to generate a CSV file (crypto_data.csv) with selected cryptocurrency symbols.
  Use sp500_scraper.py to generate a CSV file (sp500_data.csv) with symbols of current S&P 500 companies.
## Clustering and Outlier Detection:
  Modify k_means_cluster.py to apply K-Means clustering to financial data.
  Use scaler.py for data scaling using StandardScaler.
## Trading Strategy:
  ### Utilize JOKR_strat.py for running the machine learning-based trading strategy:
    Adjust hour_to_trade and minute_to_trade for setting the time to check allocations.
    Set crypto to False for S&P 500 companies or True for cryptocurrencies.
    Modify num_clusters to specify the number of clusters for K-Means.
## Alpaca API Utilities:
  ### Use alpaca_utils.py for interacting with the Alpaca API:
    Retrieve account information, execute trades, and set up portfolio clusters for trading strategies.
## Data Visualization:
  Explore different visualization functions in plotting.py to analyze data and trading performance.
## Web Interface:
  Run app.py to start the Flask application.
  Access http://localhost:5000/ to view the portfolio overview and positions using the index.html interface.
## AWS DynamoDB Integration:
  Use dynamo.py to manage interactions with AWS DynamoDB, including uploading portfolio metrics such as total account value and current positions.
# Instructions
  Execute the scripts (JOKR_strat.py, crypto_scraper.py, etc.) to perform data analysis, clustering, and automated trading tasks as described above.
  Start the Flask application (app.py) to serve the web interface for portfolio visualization and DynamoDB integration.
