# SANTY Crush AI-ML Model

A personal ML observation tracker and AI advisor for daily interactions, eye contact forecasting, and dress color predictions.

### How to Run

Just run the script (it'll auto-install scikit-learn & joblib on first launch if you don't have them):

```bash
python crush.py
```

### Features

- `[1] Predict` - Forecasts future eye-contact likelihood based on past weekday patterns.
- `[2] View data` - Shows logged observation history.
- `[3] Add entry` - Record looks for today or a specific date (`<date>, <count>`).
- `[4] Ask Perseus` - Personal AI advisor powered by NVIDIA NIM (prompts for API key on first use if not set).
- `[5] Approach her` - Predict her next dress color and take a quick 5-question situational quiz for approach readiness.

### Setup & Customization

Open `crush.py` and tweak the config at the top:
- `SYS_PROMPT` - Change Perseus's instructions or background context.
- `ML_DATA` - Add any past observation tuples `(month, day, situation, looks)` to seed the model.
- `NIM_API_KEY` - Set in your terminal (`$env:NIM_API_KEY="your-key"`) or paste it right into the terminal when asked.
