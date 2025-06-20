#!/usr/bin/env python3
"""
Flask dashboard for crypto price predictions
Visualizes actual vs predicted prices with Chart.js
"""

from flask import Flask, render_template, jsonify, request
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging

# Setup a basic logger for config loading, Flask's app.logger can be used later
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "crypto_ohlcv.db"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
DEFAULT_DEBUG = True

def load_app_config():
    """Loads configuration from YAML file, with fallbacks to defaults."""
    config_file_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    cfg = {
        "db_path": str(DEFAULT_DB_PATH),
        "symbols": DEFAULT_SYMBOLS,
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "debug": DEFAULT_DEBUG,
    }

    if not config_file_path.exists():
        logger.warning(f"Config file not found at {config_file_path}. Using default dashboard settings.")
        return cfg

    try:
        with open(config_file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if yaml_config:
            cfg["db_path"] = yaml_config.get("database", {}).get("ohlcv_path", cfg["db_path"])
            cfg["symbols"] = yaml_config.get("data", {}).get("symbols", cfg["symbols"])
            
            dashboard_config = yaml_config.get("dashboard", {})
            cfg["host"] = dashboard_config.get("host", cfg["host"])
            cfg["port"] = dashboard_config.get("port", cfg["port"])
            cfg["debug"] = dashboard_config.get("debug", cfg["debug"])

            # Log loaded config values or which ones are using defaults
            for key, default_val in [("db_path", str(DEFAULT_DB_PATH)), ("symbols", DEFAULT_SYMBOLS),
                                     ("host", DEFAULT_HOST), ("port", DEFAULT_PORT), ("debug", DEFAULT_DEBUG)]:
                is_default = False
                if key == "db_path":
                    is_default = cfg[key] == default_val and yaml_config.get("database", {}).get("ohlcv_path") is None
                elif key == "symbols":
                    is_default = cfg[key] == default_val and yaml_config.get("data", {}).get("symbols") is None
                elif key in ["host", "port", "debug"]:
                    is_default = cfg[key] == default_val and dashboard_config.get(key) is None
                
                if is_default:
                    logger.warning(f"{key} not found in config or its parent key is missing. Using default: {cfg[key]}")
                else:
                    logger.info(f"Loaded {key} from config: {cfg[key]}")
        else:
            logger.warning(f"Config file {config_file_path} is empty. Using default dashboard settings.")
            
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {config_file_path}: {e}. Using default dashboard settings.")
    except Exception as e:
        logger.error(f"Unexpected error loading config file {config_file_path}: {e}. Using default dashboard settings.")
        
    return cfg

app_config_values = load_app_config()

app = Flask(__name__)

# Configuration from loaded values
DB_PATH = Path(app_config_values['db_path'])
SYMBOLS = app_config_values['symbols']

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(str(DB_PATH)) # Ensure DB_PATH is a string
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', symbols=SYMBOLS)

@app.route('/api/actual_prices/<symbol>')
def get_actual_prices(symbol):
    """Get actual price data for a symbol"""
    try:
        hours = request.args.get('hours', 24, type=int)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        conn = get_db_connection()
        query = """
            SELECT timestamp, close as price
            FROM ohlcv 
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp, end_timestamp))
        conn.close()
        
        if df.empty:
            return jsonify([])
        
        # Convert to Chart.js format
        data = []
        for _, row in df.iterrows():
            data.append({
                'x': int(row['timestamp']),
                'y': float(row['price'])
            })
        
        return jsonify(data)
        
    except Exception as e:
        app.logger.error(f"Error getting actual prices for {symbol}: {e}")
        return jsonify([]), 500

@app.route('/api/predictions/<symbol>')
def get_predictions(symbol):
    """Get prediction data for a symbol"""
    try:
        hours = request.args.get('hours', 24, type=int)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        conn = get_db_connection()
        query = """
            SELECT timestamp, prediction_5min, prediction_10min, prediction_30min,
                   confidence_up, confidence_down
            FROM predictions 
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp, end_timestamp))
        conn.close()
        
        if df.empty:
            return jsonify({})
        
        # Convert to Chart.js format
        predictions_5min = []
        predictions_10min = []
        predictions_30min = []
        confidence_data = []
        
        for _, row in df.iterrows():
            timestamp = int(row['timestamp'])
            
            predictions_5min.append({
                'x': timestamp,
                'y': float(row['prediction_5min'])
            })
            
            predictions_10min.append({
                'x': timestamp,
                'y': float(row['prediction_10min'])
            })
            
            predictions_30min.append({
                'x': timestamp,
                'y': float(row['prediction_30min'])
            })
            
            confidence_data.append({
                'x': timestamp,
                'up': float(row['confidence_up']),
                'down': float(row['confidence_down'])
            })
        
        return jsonify({
            'predictions_5min': predictions_5min,
            'predictions_10min': predictions_10min,
            'predictions_30min': predictions_30min,
            'confidence': confidence_data
        })
        
    except Exception as e:
        app.logger.error(f"Error getting predictions for {symbol}: {e}")
        return jsonify({}), 500

@app.route('/api/summary/<symbol>')
def get_summary(symbol):
    """Get summary statistics for a symbol"""
    try:
        conn = get_db_connection()
        
        # Get latest actual price
        actual_query = """
            SELECT close, timestamp
            FROM ohlcv 
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """
        actual_result = conn.execute(actual_query, (symbol,)).fetchone()
        
        # Get latest prediction
        pred_query = """
            SELECT prediction_5min, prediction_10min, prediction_30min,
                   confidence_up, confidence_down, timestamp
            FROM predictions 
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """
        pred_result = conn.execute(pred_query, (symbol,)).fetchone()
        
        # Get 24h statistics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        start_timestamp = int(start_time.timestamp() * 1000)
        
        stats_query = """
            SELECT 
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(close) as avg_price,
                COUNT(*) as data_points
            FROM ohlcv 
            WHERE symbol = ? AND timestamp >= ?
        """
        stats_result = conn.execute(stats_query, (symbol, start_timestamp)).fetchone()
        
        conn.close()
        
        summary = {
            'symbol': symbol,
            'latest_price': float(actual_result['close']) if actual_result else None,
            'latest_price_time': actual_result['timestamp'] if actual_result else None,
            'latest_prediction': {
                '5min': float(pred_result['prediction_5min']) if pred_result else None,
                '10min': float(pred_result['prediction_10min']) if pred_result else None,
                '30min': float(pred_result['prediction_30min']) if pred_result else None,
                'confidence_up': float(pred_result['confidence_up']) if pred_result else None,
                'confidence_down': float(pred_result['confidence_down']) if pred_result else None,
                'timestamp': pred_result['timestamp'] if pred_result else None
            },
            '24h_stats': {
                'min_price': float(stats_result['min_price']) if stats_result['min_price'] else None,
                'max_price': float(stats_result['max_price']) if stats_result['max_price'] else None,
                'avg_price': float(stats_result['avg_price']) if stats_result['avg_price'] else None,
                'data_points': int(stats_result['data_points']) if stats_result['data_points'] else 0
            }
        }
        
        return jsonify(summary)
        
    except Exception as e:
        app.logger.error(f"Error getting summary for {symbol}: {e}")
        return jsonify({}), 500

@app.route('/api/accuracy/<symbol>')
def get_accuracy(symbol):
    """Calculate prediction accuracy for a symbol"""
    try:
        hours = request.args.get('hours', 24, type=int)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        conn = get_db_connection()
        
        # Get predictions and actual prices with time alignment
        query = """
            SELECT 
                p.timestamp as pred_timestamp,
                p.prediction_5min,
                p.prediction_10min,
                p.prediction_30min,
                a5.close as actual_5min,
                a10.close as actual_10min,
                a30.close as actual_30min
            FROM predictions p
            LEFT JOIN ohlcv a5 ON a5.symbol = p.symbol 
                AND a5.timestamp BETWEEN p.timestamp + 4*60*1000 AND p.timestamp + 6*60*1000
            LEFT JOIN ohlcv a10 ON a10.symbol = p.symbol 
                AND a10.timestamp BETWEEN p.timestamp + 9*60*1000 AND p.timestamp + 11*60*1000
            LEFT JOIN ohlcv a30 ON a30.symbol = p.symbol 
                AND a30.timestamp BETWEEN p.timestamp + 29*60*1000 AND p.timestamp + 31*60*1000
            WHERE p.symbol = ? AND p.timestamp >= ? AND p.timestamp <= ?
            ORDER BY p.timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, start_timestamp, end_timestamp))
        conn.close()
        
        if df.empty:
            return jsonify({'error': 'No data available for accuracy calculation'})
        
        # Calculate accuracy metrics
        accuracy = {}
        
        for timeframe in ['5min', '10min', '30min']:
            pred_col = f'prediction_{timeframe}'
            actual_col = f'actual_{timeframe}'
            
            # Filter out rows with missing actual values
            valid_data = df.dropna(subset=[pred_col, actual_col])
            
            if len(valid_data) > 0:
                # Mean Absolute Error
                mae = abs(valid_data[pred_col] - valid_data[actual_col]).mean()
                
                # Mean Absolute Percentage Error
                mape = (abs(valid_data[pred_col] - valid_data[actual_col]) / valid_data[actual_col] * 100).mean()
                
                # Root Mean Square Error
                rmse = ((valid_data[pred_col] - valid_data[actual_col]) ** 2).mean() ** 0.5
                
                accuracy[timeframe] = {
                    'mae': float(mae),
                    'mape': float(mape),
                    'rmse': float(rmse),
                    'sample_size': len(valid_data)
                }
            else:
                accuracy[timeframe] = {
                    'mae': None,
                    'mape': None,
                    'rmse': None,
                    'sample_size': 0
                }
        
        return jsonify(accuracy)
        
    except Exception as e:
        app.logger.error(f"Error calculating accuracy for {symbol}: {e}")
        return jsonify({}), 500

# Create templates directory and HTML template
def create_templates():
    """Create templates directory and HTML file"""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Price Prediction Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@2.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .symbol-tabs { display: flex; justify-content: center; margin-bottom: 20px; }
        .tab { padding: 10px 20px; margin: 0 5px; background: #ddd; border: none; cursor: pointer; border-radius: 5px; }
        .tab.active { background: #007bff; color: white; }
        .charts-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { margin-bottom: 15px; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { font-size: 1.2em; color: #333; }
        .confidence-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; margin-top: 5px; }
        .confidence-fill { height: 100%; transition: width 0.3s ease; }
        .confidence-up { background: #28a745; }
        .confidence-down { background: #dc3545; }
        .accuracy-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px; }
        .accuracy-item { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Crypto Price Prediction Dashboard</h1>
            <p>Real-time predictions vs actual prices</p>
        </div>
        
        <div class="symbol-tabs">
            {% for symbol in symbols %}
            <button class="tab {% if loop.first %}active{% endif %}" onclick="switchSymbol('{{ symbol }}')">{{ symbol }}</button>
            {% endfor %}
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h3>Price Chart & Predictions</h3>
                <canvas id="priceChart" width="800" height="400"></canvas>
            </div>
            
            <div class="summary-container">
                <h3>Current Status</h3>
                <div id="summaryContent">Loading...</div>
                
                <h4>Prediction Accuracy (24h)</h4>
                <div id="accuracyContent">Loading...</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Confidence Levels</h3>
            <canvas id="confidenceChart" width="800" height="200"></canvas>
        </div>
    </div>

    <script>
        let currentSymbol = '{{ symbols[0] }}';
        let priceChart, confidenceChart;
        
        function switchSymbol(symbol) {
            currentSymbol = symbol;
            
            // Update tab appearance
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Refresh data
            loadData();
        }
        
        function initCharts() {
            // Price chart
            const priceCtx = document.getElementById('priceChart').getContext('2d');
            priceChart = new Chart(priceCtx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Actual Price',
                            data: [],
                            borderColor: '#007bff',
                            backgroundColor: 'rgba(0, 123, 255, 0.1)',
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: '5min Prediction',
                            data: [],
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            fill: false,
                            tension: 0.1,
                            pointRadius: 3
                        },
                        {
                            label: '10min Prediction',
                            data: [],
                            borderColor: '#ffc107',
                            backgroundColor: 'rgba(255, 193, 7, 0.1)',
                            fill: false,
                            tension: 0.1,
                            pointRadius: 3
                        },
                        {
                            label: '30min Prediction',
                            data: [],
                            borderColor: '#dc3545',
                            backgroundColor: 'rgba(220, 53, 69, 0.1)',
                            fill: false,
                            tension: 0.1,
                            pointRadius: 3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'hour'
                            }
                        },
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
            
            // Confidence chart
            const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
            confidenceChart = new Chart(confidenceCtx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Confidence Up',
                            data: [],
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.2)',
                            fill: true,
                            tension: 0.1
                        },
                        {
                            label: 'Confidence Down',
                            data: [],
                            borderColor: '#dc3545',
                            backgroundColor: 'rgba(220, 53, 69, 0.2)',
                            fill: true,
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'hour'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
        }
        
        async function loadData() {
            try {
                // Load actual prices
                const actualResponse = await fetch(`/api/actual_prices/${currentSymbol}?hours=24`);
                const actualData = await actualResponse.json();
                
                // Load predictions
                const predResponse = await fetch(`/api/predictions/${currentSymbol}?hours=24`);
                const predData = await predResponse.json();
                
                // Load summary
                const summaryResponse = await fetch(`/api/summary/${currentSymbol}`);
                const summaryData = await summaryResponse.json();
                
                // Load accuracy
                const accuracyResponse = await fetch(`/api/accuracy/${currentSymbol}?hours=24`);
                const accuracyData = await accuracyResponse.json();
                
                // Update charts
                updateCharts(actualData, predData);
                updateSummary(summaryData);
                updateAccuracy(accuracyData);
                
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        function updateCharts(actualData, predData) {
            // Update price chart
            priceChart.data.datasets[0].data = actualData;
            priceChart.data.datasets[1].data = predData.predictions_5min || [];
            priceChart.data.datasets[2].data = predData.predictions_10min || [];
            priceChart.data.datasets[3].data = predData.predictions_30min || [];
            priceChart.update();
            
            // Update confidence chart
            const confidenceUp = (predData.confidence || []).map(item => ({x: item.x, y: item.up}));
            const confidenceDown = (predData.confidence || []).map(item => ({x: item.x, y: item.down}));
            
            confidenceChart.data.datasets[0].data = confidenceUp;
            confidenceChart.data.datasets[1].data = confidenceDown;
            confidenceChart.update();
        }
        
        function updateSummary(data) {
            const summaryContent = document.getElementById('summaryContent');
            
            if (!data.latest_price) {
                summaryContent.innerHTML = '<p>No data available</p>';
                return;
            }
            
            const latestPred = data.latest_prediction;
            const stats = data['24h_stats'];
            
            summaryContent.innerHTML = `
                <div class="metric">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">$${data.latest_price.toFixed(2)}</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">24h Range</div>
                    <div class="metric-value">$${stats.min_price?.toFixed(2) || 'N/A'} - $${stats.max_price?.toFixed(2) || 'N/A'}</div>
                </div>
                
                ${latestPred.timestamp ? `
                <div class="metric">
                    <div class="metric-label">Next Predictions</div>
                    <div class="metric-value">
                        5min: $${latestPred['5min']?.toFixed(2) || 'N/A'}<br>
                        10min: $${latestPred['10min']?.toFixed(2) || 'N/A'}<br>
                        30min: $${latestPred['30min']?.toFixed(2) || 'N/A'}
                    </div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Confidence</div>
                    <div>
                        <div>Up: ${(latestPred.confidence_up * 100).toFixed(1)}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill confidence-up" style="width: ${latestPred.confidence_up * 100}%"></div>
                        </div>
                        <div>Down: ${(latestPred.confidence_down * 100).toFixed(1)}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill confidence-down" style="width: ${latestPred.confidence_down * 100}%"></div>
                        </div>
                    </div>
                </div>
                ` : '<p>No recent predictions</p>'}
            `;
        }
        
        function updateAccuracy(data) {
            const accuracyContent = document.getElementById('accuracyContent');
            
            if (!data || Object.keys(data).length === 0) {
                accuracyContent.innerHTML = '<p>No accuracy data available</p>';
                return;
            }
            
            accuracyContent.innerHTML = `
                <div class="accuracy-grid">
                    ${Object.entries(data).map(([timeframe, metrics]) => `
                        <div class="accuracy-item">
                            <strong>${timeframe}</strong><br>
                            MAPE: ${metrics.mape ? metrics.mape.toFixed(2) + '%' : 'N/A'}<br>
                            Samples: ${metrics.sample_size}
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            loadData();
            
            // Refresh data every 30 seconds
            setInterval(loadData, 30000);
        });
    </script>
</body>
</html>
    """
    
    template_file = templates_dir / "dashboard.html"
    with open(template_file, 'w') as f:
        f.write(html_content)

# Create templates on startup
create_templates()

if __name__ == '__main__':
    app.run(
        host=app_config_values['host'], 
        port=app_config_values['port'], 
        debug=app_config_values['debug']
    )