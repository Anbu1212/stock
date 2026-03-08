# Flask UI for Stock Market Prediction 🔧

## What I added
- A simple Flask UI (`app.py` + `templates/index.html`) that shows next-day predictions for RNN and LSTM.
- Robust startup behavior: `app.py` now **gracefully falls back** to a safe default (last observed price) if the trained model files (`rnn_model.h5`, `lstm_model.h5`) or `scaler.pkl` are missing or incompatible with your environment.
- A helper script `prepare_models.py` that can generate and save lightweight models and a scaler (requires a compatible TensorFlow / h5py / numpy environment).

## How to run
1. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Generate the model files (if you have a compatible environment):
   ```bash
   python prepare_models.py
   ```
   This will create `rnn_model.h5`, `lstm_model.h5`, and `scaler.pkl` in the project root.

3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Notes
- If TensorFlow/Keras cannot be imported (common when numpy/h5py versions conflict), the server will still start and display fallback predictions (last observed open price).
- No changes were made to the model training logic files; I only added UI-related robustness and helper scripts so the app runs reliably in various environments.

If you'd like, I can:
- Improve the UI with a form, charts, or styling.
- Help make `prepare_models.py` run in your environment (pin working versions of TensorFlow/numpy/h5py).
