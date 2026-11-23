# CART 484 Accessible Weather Station — Version 3.0

## What's new in Version 3.0

The Raspberry Pi forecasts the next 12 hours and summarizes the next three calendar days offline. Its only sensor inputs are **station pressure (kPa), air temperature (°C) and relative humidity (%)**. The timestamps supply the hour of day and season.

- **Short-term temperature.** The model combines changes in the three readings over the past 72 hours with the time of day and season. It uses the new regression for hours 1–12, then gradually shifts to the previous regression by hour 24.
- **Daily minimum temperature.** The model now learns each day's low directly from the lowest temperature observed on that calendar day. Daily highs still come from the hourly temperature curve.
- **Pi memory use.** The model stores its coefficients as 32-bit numbers and reads the sensor CSV a row at a time. The Pi runs the trained model using only Python's standard library.
- **Measured improvement.** Across 152 historical Montréal airport forecast times, mean absolute error fell from 1.72 to 1.52 °C for the first 12 hours and from 3.90 to 3.67 °C for the third day's low. The third day's high averaged 3.60 °C of error in both versions. See the temperature backtests below for dates and methods.

Copy the **pi** folder to Jimmy's Raspberry Pi. It contains the trained offline model, an example and a memory check. The root-level forecast command also uses the upgraded model. The Pi runtime uses Python 3.8+.

## Run on the Pi

From the extracted package on the development computer, copy the folder to the Pi. Substitute the account and hostname if needed.

```sh
scp -r pi admin@meteo.local:~/weather-model-v3
ssh admin@meteo.local
cd ~/weather-model-v3
free -m
python3 forecast.py example_input.csv
python3 check_memory.py
```

Copy to a fresh destination when using this example. The sample replays the airport readings ending November 10, 2026 at 23:00 fixed UTC−5. Its JSON output is in example_forecast.json. For a live outdoor sensor CSV, use `python3 forecast.py sensor_hourly.csv --live`. The forecast runtime requires only forecast.py, station_features.py and weights.bin.gz. Python's standard library handles all inference.

On an x86-64 Linux host running Python 3.12, the shipped model occupied 70,443 bytes on disk. Under a 48 MiB per-process virtual-address limit, the standalone CLI peaked at 10.5 MiB resident memory with the sample. Replaying the entire 33,795-row CSV also used about 10.5 MiB. The uploaded CLI used about 10.9 MiB with the same sample. A separate warm run of 100 forecasts including feature construction and output formatting as Python objects used 11.1 MiB peak RSS. Check `upgrade/benchmark.json` for its timings. The original and upgraded model can both fit comfortably if 100 MiB is available to the application. If the board has 100 MiB of total RAM, check the **available** column in `free -m` with the normal web and sensor services running. Device RAM, ARM speed, whole-service memory and battery life require measurements on the actual Pi.

## Connect the website

Import `load_model` and `forecast` from the deployed forecast.py. Load the weights once per worker and cache one result per new hourly sample. The normal web request can serve the cached JSON. Read from the existing sensor database so the forecast shares the device's sampling schedule.

- Query the latest 73 hourly slots. Supply a row at the latest clock hour and at offsets 1, 3, 6, 12, 24, 48 and 72 hours before it. The app can select the sample at the hour, or the most recent sample at or before it within the five-minute sampling interval. Mark the hour missing if none qualifies.
- Use the columns time_lst, temperature_c, humidity_pct and pressure_kpa. Supply station pressure. Convert sensor hPa to kPa once by dividing by 10. Normalize timestamps to fixed UTC−5 year-round, including Montréal summer time.
- Pass a naive datetime in fixed UTC−5 as reference_time on live calls. The model rejects future/stale latest observations, missing required lags, and non-finite sensor values. Label insufficient history as collecting data. Cached responses should still be checked for freshness.
- The JSON supplies next_12_hours and next_3_days with the same field names as the upload. Each daily summary represents a complete future calendar day. Values on an indoor or unshielded sensor may differ substantially from these airport backtests.

The supplied archive contains the inference module. Jimmy still needs to wire the site's Flask route to it and verify the installed outdoor sensor. Power and accuracy on the assembled station have not been measured.

## Forecast model details

The temperature regression uses the same eight sensor timestamps. New features combine time of day and season with observed temperature, humidity, pressure and their changes. Training uses hourly issuance times. The two recipes share a September 1, 2026 training-label cutoff. The uploaded model used 4,703 six-hourly training origins, and the new temperature fit uses 28,661 hourly origins from the same observation record. Adjacent origins share data.

The short-range model provides hours 1–12 and blends linearly into the original hourly model by hour 24. Later hourly temperatures and the humidity and pressure curves use the original regression. Daily lows are learned directly from observed calendar-day minima. Daily highs use the maximum of the original hourly temperature curve. When tomorrow begins within the next 12 hours, the published daily range includes the visible hourly estimates. Direct daily lows can fall below every point on the average hourly curve.

Weights are stored as packed 32-bit floats. The CLI streams CSV rows and bounds the hourly history in memory. Sky estimates use the previous French humidity/pressure rules. There is no new precipitation or cloud accuracy result.

## Temperature backtests

The archive identifies Environment and Climate Change Canada's MONTREAL INTL A station (ID 51157) as the source of 33,795 valid hourly observations from January 1, 2023 through November 10, 2026. Raw monthly files and download checksums are included. No additional source was introduced in the upgrade.

The table compares the two exported models on all 152 original six-hourly forecast origins from October 1 through November 7, 2026. Errors are mean absolute errors in degrees Celsius. Displayed daily summaries are scored at their published precision. Hourly temperatures are scored before rounding. Oct–Nov results informed the decision about which candidate outputs to deploy, so this comparison is a retrospective development benchmark.

| Forecast | Previous version | Version 3.0 |
|---|---:|---:|
| Hours 1–12 | 1.72 | 1.52 |
| Hours 1–24 | 2.21 | 2.08 |
| Hours 25–48 | 3.43 | 3.43 |
| Hours 49–72 | 3.60 | 3.60 |
| Tomorrow low / high | 3.25 / 3.03 | 2.67 / 3.02 |
| Second calendar day low / high | 3.91 / 3.54 | 3.54 / 3.54 |
| Third calendar day low / high | 3.90 / 3.60 | 3.67 / 3.60 |

For a separate retrospective check of the recipes, both were refit with all training labels before January 1, 2025, then evaluated at 973 six-hourly origins from January through September 2025. Hour 1–12 MAE went from 1.74 to 1.58 °C. Third-calendar-day low MAE went from 3.64 to 3.30 °C and high MAE was 4.15 °C for both. That check used earlier fitted weights. Forecast windows overlap. The mean error can hide misses larger than 6 °C on individual occasions.

## Reproduce on a development computer

The bundled CSV and original weights are sufficient. Install NumPy on the development computer (tested with 2.3.5). From the root run:

```sh
python upgrade/train_upgrade.py
python upgrade/season_check.py
python upgrade/verify_runtime.py
```

The training script chooses ridge penalties from four expanding-window validation blocks ending before October 2026, exports the model, and compares its portable outputs with the fitted arrays on every original test origin. The earlier-season script refits both recipes for its historical check. The runtime checks cover every issuance hour, calendar dates, public daily ranges, stale or incomplete history, the vector API and streamed CSV replay.

Results for this release are in upgrade/evaluation.json, upgrade/season_check.json and upgrade/benchmark.json. The root metrics.json, model.json.gz, train.py, verify.py and README-baseline.md describe the original model. Its runtime is baseline_forecast.py. Root forecast.py accepts an explicit legacy JSON gzip model path for callers replaying that format.

## From Sensor Readings to the Website

1. **Train the model on a development computer.** We used historical hourly weather observations from Environment and Climate Change Canada’s Montréal airport station. The model learns how recent changes in temperature, relative humidity, and station pressure, together with the time of day and season, relate to what happens over the following hours and days. The fitted coefficients are saved in `pi/weights.bin.gz`.

2. **Read the station’s own sensor history.** Once the weather station is running, the forecast uses its outdoor sensor readings from the existing database. Each forecast uses eight hourly readings covering the latest hour and 1, 3, 6, 12, 24, 48, and 72 hours before it. If a required reading is missing or the latest reading is stale, the website should show that the forecast is unavailable. During startup, it can show “Collecting data” while the station builds up enough history.

3. **Calculate the forecast on the Raspberry Pi.** The Pi applies the saved coefficients locally using Python’s standard library. It returns hourly estimates for the next 12 hours and a summary for each of the next three full calendar days. The sky descriptions come from humidity and pressure rules. Forecasting runs offline, with training done separately on the development computer.

4. **Serve the forecast through Flask.** Copy the `pi/` folder to the Raspberry Pi and run `python3 forecast.py example_input.csv` from that folder to check the sample output. Then connect the existing Flask app to the forecasting module and sensor database. Load the model once per Flask worker, refresh the forecast when a new hourly reading is available, and serve the cached `next_12_hours` and `next_3_days` data to the website.

The package includes the trained model and a working command-line example. The remaining integration work is to connect the Flask route and test it with the actual outdoor sensor. Caching keeps page loads from triggering a new forecast. Battery impact still needs to be measured on the assembled station.