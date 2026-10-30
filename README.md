# S4NTY Crush AI & ML Observation Model

A terminal-based AI & ML observation tracker that predicts eye-contact likelihood using Decision Trees, forecasts daily dress colors, and provides grounded AI advice.

## Quick Start

Run the standalone script:
```bash
python crush.py
```
*(Dependencies `scikit-learn` and `joblib` are auto-installed on first launch if missing).*

## Main Features

- `[1] Predict`: Forecast interaction probability for today or tomorrow.
- `[2] View data`: Review chronological observation logs.
- `[3] Add entry`: Record daily eye contact (`<date>, <count>`).
- `[4] Ask Perseus`: Consult the AI advisor for grounded feedback.
- `[5] Approach her`: Predict her next dress color and get approach readiness scoring.

## Customization

Open `crush.py` and customize the variables at the top of the file:
- **`SYS_PROMPT`**: Replace `{sys_prompt}` with your custom persona or instructions.
- **`ML_DATA`**: Add past observation tuples `(month, day, situation, looks)` to pre-train.
- **API Key**: Set `NIM_API_KEY` in your environment, or paste it in the CLI when prompted.
