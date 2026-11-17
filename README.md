# TEMPO reproducibility materials

This repository contains the implementation, exact released inputs, model prompts, cached model responses, and reports for **TEMPO: Temporal Evidence and Momentum for Market-Neutral Hedging** (FinReason Cup 2026, Task 2). The English manuscript is in `Docs/task2_working_note/`.

## Contents

- `task2_factor_agent.py`: event, relationship, and feedback factors; decay kernels; prompts; decision loop.
- `task2_execution_policy.py`: momentum combination, stock selection, cost threshold, and ablation policies.
- `task2_experiment.py`, `task2_hourly_validation.py`: official release parsing and pre-competition hourly calibration.
- `task2_factor_experiment.py`, `task2_return_experiment.py`: parameter and execution comparisons.
- `task2_factor_research.py`: numerical checks for the factor forecast and calibration analysis.
- `task2_factor_agent_audit.py`: release chronology, factor, decision, and return checks.
- `task2_market_beta.py`: historical SPY beta calculation and pair exposure summary.
- `research_import/task2_dataset/`: the 41 historical public price and news releases and `releases.json`.
- `research_import/task2_live_dataset/`: the historical releases plus the six Phrase III releases and their final `releases.json`.
- `research_import/task2_yahoo_1h/`: ten stock series and SPY hourly bars used for calibration and beta measurement.
- `research_import/task2_live_state.json`, `research_import/task2_combined_live_state.json`: official historical schedule and the 47-round combined schedule used in the live replay.
- `outputs/task2_factor_calls/`: saved DeepSeek responses, including request hashes and parsed output. The request messages and model settings are defined in `task2_factor_agent.py`.
- `outputs/*.json`: reported historical, live, ablation, and market-beta results.

The official data source is [Market-Neutral Hedging live trading](https://huggingface.co/datasets/CarolynJiang/Market-Neutral-Hedging-live-trading). The archived Yahoo chart response URLs follow `https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}?interval=60m&range=730d&includePrePost=false` for the ten stocks and SPY. Each market-beta report records the SHA-256 of the SPY input. The market proxy is SPY, and all beta calibration observations precede the first competition release.

## Reproduce the results

Use Python 3.11 or newer in the repository root. Install the versions in `requirements-task2.txt`. The archived model responses let these commands replay without a DeepSeek API key. If a required response is absent or its request hash differs, the agent stops at that request and requires `DEEPSEEK_API_KEY` for a new call.

```powershell
python -m pip install -r requirements-task2.txt
python task2_factor_agent.py --limit 41 --output outputs/task2_factor_agent_replay.json
python task2_factor_agent_audit.py --report outputs/task2_factor_agent_replay.json
python task2_factor_agent.py --data research_import/task2_live_dataset --state research_import/task2_combined_live_state.json --limit 47 --output outputs/task2_factor_agent_live_replay.json
python task2_factor_agent_audit.py --data research_import/task2_live_dataset --state research_import/task2_combined_live_state.json --report outputs/task2_factor_agent_live_replay.json
python task2_return_experiment.py
python task2_market_beta.py
python -m pytest tests -q
```

The historical replay contains 41 releases and 37 evaluable rounds. The live replay appends six independent rounds; the official live return is reported in the manuscript and is separate from the historical cumulative return. `task2_market_beta.py` fits stock betas on 3,490 matched historical intraday intervals, then computes each equal-dollar long–short pair's estimated beta as half the long-stock beta minus half the short-stock beta.

The manuscript source uses IEEEtran and BibTeX. Build the PDF with `latexmk -pdf working_note.tex` from `Docs/task2_working_note/`.
