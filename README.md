# CART 484 — Local weather model

This package contains a trained forecasting module, not a completed Flask integration. It produces the next 12 hourly estimates and summaries for the next three complete calendar days. It does not modify the website, read the BME680, or start another sensor process.

## Quick demonstration

Python 3 standard library only; no internet, Ollama, numpy or scikit-learn is needed for inference:

```sh
python3 forecast.py example_input.csv
```

This replays historical observations ending at 2026-11-10 23:00 local standard time. It is NOT a current forecast. `example_forecast.json` contains the corresponding output.

For live operation use `--live`, or pass the current fixed UTC-5 naive datetime as `reference_time` to `forecast(rows, model, reference_time)`. The function then rejects future inputs or an observation more than 90 minutes old. Missing required history is also rejected.

## Installation on Jimmy's Raspberry Pi

The runtime needs only `forecast.py`, `features.py`, and `model.json.gz`. Include `example_input.csv` for an initial test. Do not copy training datasets to the Pi unless you want them there.

1. Extract this package on your computer.
2. From its directory, create a destination and copy these files (replace the hostname if your Pi uses another address):

```sh
ssh admin@meteo.local "mkdir -p /home/admin/weather-model"
scp forecast.py features.py model.json.gz example_input.csv admin@meteo.local:/home/admin/weather-model/
ssh admin@meteo.local
cd /home/admin/weather-model
python3 forecast.py example_input.csv
```

The computer must be able to reach the Pi, for example by connecting to its hotspot. The historical demonstration should print JSON with `next_12_hours` and `next_3_days`.

**Copying these files does not connect them to the website.** The remaining Flask integration must:

- Load the model once when the application starts, using `load_model()`.
- Read existing database observations; do not open a second BME680 reader.
- Convert five-minute records into consistently sampled hourly observations. Use the same hourly sampling convention as the training data, rather than treating adjacent five-minute readings as hours.
- Supply `time_lst`, `temperature_c`, `humidity_pct`, and `pressure_kpa`. Timestamps must be fixed UTC-5 standard time, not daylight-saving wall time. Convert raw hPa to kPa by dividing by 10 exactly once.
- Require observations at the current hour and 1, 3, 6, 12, 24, 48, and 72 hours earlier. Missing history must show a collecting/unavailable state or an explicitly labelled rule-based fallback.
- Call the model when a new hourly input is available, cache its result, and map the output into the website's existing API/frontend fields. The output is compatible with the previous standalone model package, not automatically with the Flask API.
- Keep actual measured values separate from predictions. Check freshness on live requests and identify experimental predictions in the interface and spoken output.
- Restart `bme680-web.service` only after that integration has been implemented and checked. Merely restarting the service after copying this standalone package will not activate the model.

The model requires 72 hours of history. Indoor readings should not be presented as outdoor forecasts: use the labelled airport demonstration until the device has suitable outdoor protection, then validate the model using its outdoor observations.

## Pi workload and battery

The selected model is a small linear Ridge model, not a language model. The compressed model is 141,207 bytes (about 138 KiB). It has 51 inputs and 288 outputs: 96 hours multiplied by three variables. A prediction uses about 14,688 coefficient multiplications plus additions, feature calculations and formatting.

Training happens on the development computer. The Pi only loads the fitted coefficients and evaluates them. No training libraries, cloud calls, or GPU are needed on the Pi. The decoded Python objects occupy more memory than the compressed file.

Expected incremental compute load is small, but **battery life, power consumption and runtime have not been measured on your Pi**. No battery-hour or percentage-saving claim is justified yet. For efficient integration, load once and reuse, calculate once per new hourly observation, and serve a cached forecast to all browsers. Keep the existing five-minute sensor collection schedule. Do not retrain or launch a new Python process for every page request.

## Data and provenance

Environment and Climate Change Canada, MONTREAL INTL A, stationID 51157, climate ID 7025251.

- January 1, 2023 through November 10, 2026: 33,795 valid hourly observations.
- Temperature in Celsius, relative humidity in percent, station pressure in kPa.
- 45 missing or invalid rows removed; no fabricated values or interpolation.
- Local STANDARD time (LST), fixed UTC-5, including summer dates.
- Downloaded November 11, 2026. The 47 raw monthly files are under `raw/`.
- `sources.json` records URLs and SHA256 hashes; `observations.csv` contains cleaned observations. The downloader retains existing files for reproducibility.

Example official download URL (change Year and Month for each file):
https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=51157&Year=2023&Month=1&Day=1&timeframe=1&submit=Download+Data

## Features, training and formulas

Candidates were Ridge with alpha 10, 100 and 1000, and a size-limited Extra Trees regressor. Selection used only September 2026 validation temperature MAE over hours 1–72. The winner was **Ridge alpha=100**. Final test results were not used to select parameters.

Inputs comprise:

- Temperature, humidity and pressure at lags 0, 1, 3, 6, 12, 24, 48, 72 hours: 24 features.
- Current values minus values at each of the seven nonzero lags: 21 features.
- sin/cos of the hour of day, plus first and second annual harmonics: 6 features.

Hour phase is `2*pi*hour/24`. Annual phase is `2*pi*(days since January 1)/365.2425`; both its first and second harmonic are included.

Standardization uses training statistics only:

    z_i = (x_i - training_mean_i) / training_scale_i
    predicted_change(h,k) = intercept(h,k) + sum(weight(h,k,i) * z_i)
    predicted_value(h,k) = current_value(k) + predicted_change(h,k)

The training objective is squared error plus `100*sum(weight^2)`; intercepts are not penalized. Exact coefficients are in `model.json.gz`.

Each future hour is predicted directly, without feeding predictions back as observations. The internal horizon is 96 hours to ensure three complete future calendar days at any issuance hour. A 72-hour horizon alone cannot cover all of the third calendar day when issued early in the current day. The interface output remains 12 hourly entries and three daily summaries.

## Chronological evaluation

One forecast origin every six hours:

- Training: 4,703 origins; all 96-hour labels are before September 1, 2026.
- Validation: 104 origins starting in September; all labels are before October 1.
- Testing: 152 origins, October 1 00:00 to November 7 18:00; observed answers for the first 72 hours extend through November 10.

Missing required history/labels is excluded. Some final test origins lack labels beyond hour 72; main metrics use complete hours 1–72. Calendar-day metrics require a complete observed day. The shipped model is the evaluated model, not a model refitted on test answers. Overlapping forecast windows are not independent evidence. Both data volume and features changed, so improvement cannot be attributed to data volume alone.

Temperature mean absolute error, Celsius:

| Horizon | New model | Current temperature unchanged | Repeat past 24-hour curve |
|---|---:|---:|---:|
| Hours 1–12 | 1.72 | 3.14 | 3.79 |
| Hours 1–24 | 2.21 | 3.69 | 3.78 |
| Hours 25–48 | 3.43 | 5.30 | 4.76 |
| Hours 49–72 | 3.60 | 5.29 | 4.15 |

These are averages, not guaranteed error limits. Individual errors can be larger. The test covers autumn, not every season.

The old model trained on part of October, so a fair old/new comparison uses only its original held-out November 4–7 window, at the same 16 six-hour origins:

| Horizon | Previous model | New model |
|---|---:|---:|
| Hours 1–12 | 1.48 | 2.00 |
| Hours 25–48 | 4.08 | 3.47 |
| Hours 49–72 | 5.85 | 2.13 |

Day-three performance improved, but short-term performance worsened in this small window. The previously reported 5.88 C used hourly origins; 5.85 C here uses the matched six-hour origins.

For the website's daily minimum/maximum temperatures, the broader test produced:

| Calendar day | Minimum MAE | Maximum MAE |
|---|---:|---:|
| Tomorrow | 3.25 | 3.03 |
| Following day | 3.91 | 3.55 |
| Third day | 3.90 | 3.60 |

Each row uses 152 complete calendar-day cases, with overlapping dates possible. Daily extrema are the minimum/maximum of 24 hourly predictions. These metrics differ from rolling 24-hour horizon errors.

## Sky descriptions remain heuristic

No rain/cloud classifier or calibrated precipitation probability has been trained. Improved temperature error does not demonstrate improved sky descriptions.

Let RH be predicted humidity, clipped to 0–100%, and dP be predicted pressure minus current pressure in kPa. Apply in order:

1. RH >= 85 and dP <= -0.3: `Risque de précipitations`.
2. Otherwise RH >= 75: `Nuageux`.
3. Otherwise RH >= 55: `Partiellement nuageux`.
4. Otherwise: `Possiblement dégagé`.

Example: RH=88%, pressure falling from 101.2 to 100.8 kPa gives dP=-0.4 and a precipitation-risk description. These are prototype thresholds, not verified rain/cloud observations. Daily descriptions use the most frequent hourly category; ties use the earliest encountered category. No percentage rain probability is produced.

## Advantages over manually chosen formulas

The model is itself a mathematical formula: its coefficients are learned from historical observations rather than manually chosen. It can combine pressure/humidity changes, multiple time scales and seasonal/daytime cycles, and can be evaluated against future observations.

The documented improvements are against persistence baselines and the previous trained model. **We have not run a matched backtest against the website's original hand-written forecasting formulas**, so we cannot claim the model is proven superior to those formulas. Rules remain easier to explain, can work with shorter history, and are useful as an explicitly labelled fallback. Neither approach can observe incoming regional weather systems from one local sensor.

## Reproduce training and checks

On the development computer:

```sh
python -m pip install -r requirements-training.txt
python download.py
python train.py
python verify.py
```

Raw data is already included; downloading is only needed for missing files. Dates, candidate models and the random seed are fixed. Update the experimental boundaries deliberately when using newer data. `previous_model.json` is the frozen old model for the matched comparison.

`verify.py` checks all 152 test origins and 96 output hours against training-library predictions, checks complete three-day output at every issuance hour, rejects missing/stale input, and calculates daily-extrema metrics. Details are in `metrics.json`, `test_predictions.csv` and `verification.npz`.
