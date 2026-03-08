from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# `load_model` is imported inside the try block to avoid importing tensorflow at module import time
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'

# Database setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, 'users.db')

def init_db():
    """Initialize database with users table (create if missing)"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    # Ensure users table exists on every connection (defensive)
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    conn.commit()
    return conn

# Initialize database on app start
init_db()

# Load models and scaler (graceful fallback if files are missing)
try:
    # Import load_model lazily to avoid heavy tensorflow import during module load (may not be available in all envs)
    from keras.models import load_model
    regressor = load_model('rnn_model.h5')
    model_lstm = load_model('lstm_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    models_loaded = True
except Exception as e:
    # If models/scaler can't be loaded (environment issues or missing files), fall back to simple predictors
    print(f"Warning: could not load models or scaler - {e}")
    regressor = None
    model_lstm = None
    scaler = None
    models_loaded = False

# Load data for prediction (default)
DEFAULT_CSV = os.path.join(BASE_DIR, 'RELIANCE.NS.csv')
try:
    data = pd.read_csv(DEFAULT_CSV)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
except Exception as e:
    print(f"Warning: could not load default CSV - {e}")
    data = pd.DataFrame()

time_step = 50

def load_data_from_session():
    """Load CSV based on session 'symbol' (filename) or fall back to default."""
    try:
        symbol = session.get('symbol', os.path.basename(DEFAULT_CSV)) if 'session' in globals() or True else os.path.basename(DEFAULT_CSV)
        path = symbol if os.path.isabs(symbol) else os.path.join(BASE_DIR, symbol)
        if not os.path.exists(path):
            path = DEFAULT_CSV
        
        df = pd.read_csv(path)
        
        # Handle Date column if it exists
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception:
                print(f"Warning: could not convert Date column to datetime")
        
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Return default data as fallback
        try:
            df = pd.read_csv(DEFAULT_CSV)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception:
            return pd.DataFrame()

def get_time_step():
    try:
        return int(session.get('time_step', time_step))
    except Exception:
        return time_step

def generate_recommendation(predicted_price, current_price):
    """Generate buy/sell/hold recommendation based on predicted vs current price"""
    percent_change = ((predicted_price - current_price) / current_price) * 100
    
    if percent_change > 2:
        return "BUY", f"Strong upward trend expected (+{percent_change:.2f}%)"
    elif percent_change > 0.5:
        return "BUY", f"Moderate upward trend expected (+{percent_change:.2f}%)"
    elif percent_change < -2:
        return "SELL", f"Strong downward trend expected ({percent_change:.2f}%)"
    elif percent_change < -0.5:
        return "SELL", f"Moderate downward trend expected ({percent_change:.2f}%)"
    else:
        return "HOLD", f"Neutral trend expected (~{percent_change:.2f}%)"

def generate_classification(predicted_price, current_price, historical_data):
    """Classify market as Bullish, Bearish, or Neutral"""
    recent_avg = historical_data[-20:].mean()
    
    if predicted_price > recent_avg and predicted_price > current_price:
        return "BULLISH", "Strong upward momentum detected"
    elif predicted_price > current_price:
        return "BULLISH", "Upward trend indicated"
    elif predicted_price < recent_avg and predicted_price < current_price:
        return "BEARISH", "Strong downward momentum detected"
    elif predicted_price < current_price:
        return "BEARISH", "Downward trend indicated"
    else:
        return "NEUTRAL", "Market showing consolidation pattern"

def generate_solution(recommendation, classification, current_price, predicted_price):
    """Generate actionable solution based on recommendation and classification"""
    solutions = []
    
    if recommendation[0] == "BUY":
        if classification[0] == "BULLISH":
            solutions.append("✓ Strong Buy Signal: Market is bullish and price expected to rise")
            solutions.append("→ Action: Consider buying at current levels for potential gains")
        else:
            solutions.append("⚠ Buy with caution: Conflicting signals detected")
            solutions.append("→ Action: Wait for confirmation before buying")
    
    elif recommendation[0] == "SELL":
        if classification[0] == "BEARISH":
            solutions.append("✓ Strong Sell Signal: Market is bearish and price expected to fall")
            solutions.append("→ Action: Consider selling to protect gains or cut losses")
        else:
            solutions.append("⚠ Sell with caution: Conflicting signals detected")
            solutions.append("→ Action: Monitor price before deciding to sell")
    
    else:  # HOLD
        solutions.append("→ Action: Hold your current position")
        solutions.append("→ Monitor: Watch for price movement confirmation")
        solutions.append("→ Threshold: Re-evaluate if price moves ±1% from current level")
    
    # Add general insight
    price_change = ((predicted_price - current_price) / current_price) * 100
    solutions.append(f"\nTarget Price: ₹{predicted_price:.2f} (Expected change: {price_change:+.2f}%)")
    
    return solutions

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('login.html')
        
        conn = None
        try:
            conn = get_db_connection()
            # Case-insensitive username lookup
            user = conn.execute('SELECT * FROM users WHERE LOWER(username) = ?', (username.lower(),)).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash('Login successful!', 'success')
                return redirect(url_for('prediction'))
            else:
                flash('Invalid username or password', 'error')
        finally:
            if conn:
                conn.close()
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password or not confirm_password:
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        # Normalize and defensive checks
        username = username.strip()
        email = email.strip().lower()

        conn = None
        try:
            conn = get_db_connection()
            # Check existing username/email case-insensitively
            existing = conn.execute('SELECT * FROM users WHERE LOWER(username)=? OR LOWER(email)=?',
                                    (username.lower(), email.lower())).fetchone()
            if existing:
                if existing['username'].lower() == username.lower():
                    flash('Username already exists', 'error')
                elif existing['email'].lower() == email.lower():
                    flash('Email already exists', 'error')
                else:
                    flash('Username or email already exists', 'error')
                return render_template('register.html')

            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, generate_password_hash(password)))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
        finally:
            if conn:
                conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip()
        time_step_val = request.form.get('time_step', '').strip()
        if symbol:
            session['symbol'] = symbol
        try:
            if time_step_val:
                session['time_step'] = int(time_step_val)
        except Exception:
            flash('Invalid time step value', 'error')
            return render_template('settings.html')
        flash('Settings saved', 'success')
        return redirect(url_for('index'))

    # Prefill with current values
    current_symbol = session.get('symbol', os.path.basename(DEFAULT_CSV))
    current_ts = session.get('time_step', time_step)
    return render_template('settings.html', symbol=current_symbol, time_step=current_ts)

@app.route('/')
def index():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Prepare data for prediction (use user-selected symbol/time_step from session)
        df = load_data_from_session()
        
        # Check if dataframe is empty or missing required columns
        if df.empty or 'Open' not in df.columns:
            flash('Error: CSV file must have an "Open" column. Please check your CSV file.', 'error')
            return redirect(url_for('settings'))
        
        ts = get_time_step()
        dataset = df.Open.values[-ts:].reshape(-1, 1)
        current_price = float(dataset[-1, 0])
        historical_prices = df.Open.values

        if models_loaded and scaler is not None and regressor is not None and model_lstm is not None:
            # Use real models
            scaled_dataset = scaler.transform(dataset)
            X_input = np.reshape(scaled_dataset, (1, time_step, 1))
            # Predictions
            try:
                rnn_prediction = scaler.inverse_transform(regressor.predict(X_input))[0, 0]
            except Exception as e:
                print(f"Warning: rnn predict failed - {e}")
                rnn_prediction = float(dataset[-1, 0])
            try:
                lstm_prediction = scaler.inverse_transform(model_lstm.predict(X_input))[0, 0]
            except Exception as e:
                print(f"Warning: lstm predict failed - {e}")
                lstm_prediction = float(dataset[-1, 0])
        else:
            # Fallback simple behavior: return last observed open price (safe default)
            last_price = float(dataset[-1, 0])
            rnn_prediction = round(last_price, 2)
            lstm_prediction = round(last_price, 2)

        # Average the predictions
        avg_prediction = (rnn_prediction + lstm_prediction) / 2
        
        # Generate recommendation based on average prediction
        recommendation = generate_recommendation(avg_prediction, current_price)
        
        # Generate classification based on market trend
        classification = generate_classification(avg_prediction, current_price, historical_prices)
        
        # Generate actionable solution
        solution = generate_solution(recommendation, classification, current_price, avg_prediction)

        return render_template('index.html', 
                             rnn_pred=round(rnn_prediction, 2), 
                             lstm_pred=round(lstm_prediction, 2),
                             avg_pred=round(avg_prediction, 2),
                             current_price=round(current_price, 2),
                             recommendation=recommendation,
                             classification=classification,
                             solution=solution)
    
    except KeyError as e:
        flash(f'Error: Missing column {e} in CSV file. Please ensure your CSV has Open column.', 'error')
        return redirect(url_for('settings'))
    except Exception as e:
        flash(f'Error loading data: {str(e)}', 'error')
        return redirect(url_for('settings'))

@app.route('/prediction')
def prediction():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Prepare data for prediction (use user-selected symbol/time_step from session)
        df = load_data_from_session()
        
        # Check if dataframe is empty or missing required columns
        if df.empty or 'Open' not in df.columns:
            flash('Error: CSV file must have an "Open" column. Please check your CSV file.', 'error')
            return redirect(url_for('settings'))
        
        ts = get_time_step()
        dataset = df.Open.values[-ts:].reshape(-1, 1)
        current_price = float(dataset[-1, 0])

        if models_loaded and scaler is not None and regressor is not None and model_lstm is not None:
            scaled_dataset = scaler.transform(dataset)
            X_input = np.reshape(scaled_dataset, (1, time_step, 1))
            try:
                rnn_prediction = scaler.inverse_transform(regressor.predict(X_input))[0, 0]
            except Exception as e:
                rnn_prediction = float(dataset[-1, 0])
            try:
                lstm_prediction = scaler.inverse_transform(model_lstm.predict(X_input))[0, 0]
            except Exception as e:
                lstm_prediction = float(dataset[-1, 0])
        else:
            last_price = float(dataset[-1, 0])
            rnn_prediction = round(last_price, 2)
            lstm_prediction = round(last_price, 2)

        avg_prediction = (rnn_prediction + lstm_prediction) / 2

        return render_template('prediction.html',
                             rnn_pred=round(rnn_prediction, 2),
                             lstm_pred=round(lstm_prediction, 2),
                             avg_pred=round(avg_prediction, 2),
                             current_price=round(current_price, 2))
    
    except KeyError as e:
        flash(f'Error: Missing column {e} in CSV file. Please ensure your CSV has Date, Open, High, Low, Close columns.', 'error')
        return redirect(url_for('settings'))
    except Exception as e:
        flash(f'Error loading prediction data: {str(e)}', 'error')
        return redirect(url_for('settings'))


@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    # Allow logged-in users only
    if 'user_id' not in session:
        return redirect(url_for('login'))

    raw = request.form.get('manual_prices', '')
    if not raw:
        flash('Please provide comma-separated prices for manual prediction', 'error')
        return redirect(url_for('index'))

    # Parse floats from input
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    try:
        manual_values = [float(p) for p in parts]
    except Exception:
        flash('Invalid input. Ensure values are numeric and comma-separated.', 'error')
        return redirect(url_for('index'))

    # Load default dataset and prepare a full dataset of length `ts`
    df = load_data_from_session()
    ts = get_time_step()
    historical = df.Open.values

    # If user provided fewer than ts values, prepend from historical data
    if len(manual_values) >= ts:
        dataset_vals = np.array(manual_values[-ts:]).reshape(-1, 1)
    else:
        needed = ts - len(manual_values)
        prefix = historical[-(needed + len(manual_values)):-len(manual_values)] if needed > 0 and len(manual_values) > 0 else historical[-needed:]
        if len(manual_values) == 0:
            dataset_vals = historical[-ts:].reshape(-1, 1)
        else:
            combined = list(prefix) + manual_values
            dataset_vals = np.array(combined[-ts:]).reshape(-1, 1)

    current_price = float(dataset_vals[-1, 0])

    # Make predictions using models if available
    if models_loaded and scaler is not None and regressor is not None and model_lstm is not None:
        try:
            scaled_dataset = scaler.transform(dataset_vals)
            X_input = np.reshape(scaled_dataset, (1, ts, 1))
            try:
                rnn_prediction = scaler.inverse_transform(regressor.predict(X_input))[0, 0]
            except Exception:
                rnn_prediction = float(dataset_vals[-1, 0])
            try:
                lstm_prediction = scaler.inverse_transform(model_lstm.predict(X_input))[0, 0]
            except Exception:
                lstm_prediction = float(dataset_vals[-1, 0])
        except Exception:
            # Scaling failed — fallback to simple behavior
            last_price = float(dataset_vals[-1, 0])
            rnn_prediction = round(last_price, 2)
            lstm_prediction = round(last_price, 2)
    else:
        last_price = float(dataset_vals[-1, 0])
        rnn_prediction = round(last_price, 2)
        lstm_prediction = round(last_price, 2)

    avg_prediction = (rnn_prediction + lstm_prediction) / 2

    recommendation = generate_recommendation(avg_prediction, current_price)
    classification = generate_classification(avg_prediction, current_price, historical)
    solution = generate_solution(recommendation, classification, current_price, avg_prediction)

    flash('Manual input used for prediction', 'success')
    return render_template('index.html',
                         rnn_pred=round(rnn_prediction, 2),
                         lstm_pred=round(lstm_prediction, 2),
                         avg_pred=round(avg_prediction, 2),
                         current_price=round(current_price, 2),
                         recommendation=recommendation,
                         classification=classification,
                         solution=solution)

@app.route('/recommendation')
def recommendation_page():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Prepare data for prediction (use user-selected symbol/time_step from session)
        df = load_data_from_session()
        
        # Check if dataframe is empty or missing required columns
        if df.empty or 'Open' not in df.columns:
            flash('Error: CSV file must have an "Open" column. Please check your CSV file.', 'error')
            return redirect(url_for('settings'))
        
        ts = get_time_step()
        dataset = df.Open.values[-ts:].reshape(-1, 1)
        current_price = float(dataset[-1, 0])

        if models_loaded and scaler is not None and regressor is not None and model_lstm is not None:
            scaled_dataset = scaler.transform(dataset)
            X_input = np.reshape(scaled_dataset, (1, time_step, 1))
            try:
                rnn_prediction = scaler.inverse_transform(regressor.predict(X_input))[0, 0]
            except Exception as e:
                rnn_prediction = float(dataset[-1, 0])
            try:
                lstm_prediction = scaler.inverse_transform(model_lstm.predict(X_input))[0, 0]
            except Exception as e:
                lstm_prediction = float(dataset[-1, 0])
        else:
            last_price = float(dataset[-1, 0])
            rnn_prediction = round(last_price, 2)
            lstm_prediction = round(last_price, 2)

        avg_prediction = (rnn_prediction + lstm_prediction) / 2
        recommendation = generate_recommendation(avg_prediction, current_price)

        return render_template('recommendation.html',
                             recommendation=recommendation,
                             avg_pred=round(avg_prediction, 2),
                             current_price=round(current_price, 2))
    
    except KeyError as e:
        flash(f'Error: Missing column {e} in CSV file. Please ensure your CSV has Open column.', 'error')
        return redirect(url_for('settings'))
    except Exception as e:
        flash(f'Error loading recommendation data: {str(e)}', 'error')
        return redirect(url_for('settings'))

@app.route('/classification')
def classification_page():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Prepare data for prediction (use user-selected symbol/time_step from session)
        df = load_data_from_session()
        
        # Check if dataframe is empty or missing required columns
        if df.empty or 'Open' not in df.columns:
            flash('Error: CSV file must have an "Open" column. Please check your CSV file.', 'error')
            return redirect(url_for('settings'))
        
        ts = get_time_step()
        dataset = df.Open.values[-ts:].reshape(-1, 1)
        current_price = float(dataset[-1, 0])
        historical_prices = df.Open.values
        historical_avg = historical_prices[-20:].mean()

        if models_loaded and scaler is not None and regressor is not None and model_lstm is not None:
            scaled_dataset = scaler.transform(dataset)
            X_input = np.reshape(scaled_dataset, (1, time_step, 1))
            try:
                rnn_prediction = scaler.inverse_transform(regressor.predict(X_input))[0, 0]
            except Exception as e:
                rnn_prediction = float(dataset[-1, 0])
            try:
                lstm_prediction = scaler.inverse_transform(model_lstm.predict(X_input))[0, 0]
            except Exception as e:
                lstm_prediction = float(dataset[-1, 0])
        else:
            last_price = float(dataset[-1, 0])
            rnn_prediction = round(last_price, 2)
            lstm_prediction = round(last_price, 2)

        avg_prediction = (rnn_prediction + lstm_prediction) / 2
        classification = generate_classification(avg_prediction, current_price, historical_prices)

        return render_template('classification.html',
                             classification=classification,
                             avg_pred=round(avg_prediction, 2),
                             current_price=round(current_price, 2),
                             historical_avg=round(historical_avg, 2))
    
    except KeyError as e:
        flash(f'Error: Missing column {e} in CSV file. Please ensure your CSV has Open column.', 'error')
        return redirect(url_for('settings'))
    except Exception as e:
        flash(f'Error loading classification data: {str(e)}', 'error')
        return redirect(url_for('settings'))

@app.route('/solution')
def solution_page():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Prepare data for prediction (use user-selected symbol/time_step from session)
        df = load_data_from_session()
        
        # Check if dataframe is empty or missing required columns
        if df.empty or 'Open' not in df.columns:
            flash('Error: CSV file must have an "Open" column. Please check your CSV file.', 'error')
            return redirect(url_for('settings'))
        
        ts = get_time_step()
        dataset = df.Open.values[-ts:].reshape(-1, 1)
        current_price = float(dataset[-1, 0])
        historical_prices = df.Open.values

        if models_loaded and scaler is not None and regressor is not None and model_lstm is not None:
            scaled_dataset = scaler.transform(dataset)
            X_input = np.reshape(scaled_dataset, (1, time_step, 1))
            try:
                rnn_prediction = scaler.inverse_transform(regressor.predict(X_input))[0, 0]
            except Exception as e:
                rnn_prediction = float(dataset[-1, 0])
            try:
                lstm_prediction = scaler.inverse_transform(model_lstm.predict(X_input))[0, 0]
            except Exception as e:
                lstm_prediction = float(dataset[-1, 0])
        else:
            last_price = float(dataset[-1, 0])
            rnn_prediction = round(last_price, 2)
            lstm_prediction = round(last_price, 2)

        avg_prediction = (rnn_prediction + lstm_prediction) / 2
        recommendation = generate_recommendation(avg_prediction, current_price)
        classification = generate_classification(avg_prediction, current_price, historical_prices)
        solution = generate_solution(recommendation, classification, current_price, avg_prediction)

        return render_template('solution.html',
                             recommendation=recommendation,
                             classification=classification,
                             solution=solution,
                             avg_pred=round(avg_prediction, 2),
                             current_price=round(current_price, 2))
    
    except KeyError as e:
        flash(f'Error: Missing column {e} in CSV file. Please ensure your CSV has Open column.', 'error')
        return redirect(url_for('settings'))
    except Exception as e:
        flash(f'Error loading solution data: {str(e)}', 'error')
        return redirect(url_for('settings'))

if __name__ == '__main__':
    app.run(debug=True)
