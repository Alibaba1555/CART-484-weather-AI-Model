"""Project entry point; the complete deployable standalone module is in pi/."""
from pathlib import Path

from pi import forecast as current
from pi.station_features import expand


def load_model(path=None):
    if path is not None and Path(path).name.endswith('.json.gz'):
        import baseline_forecast
        return baseline_forecast.load_model(path)
    return current.load_model(path)


def forecast(rows, model, reference_time=None):
    if isinstance(model, dict):
        import baseline_forecast
        return baseline_forecast.forecast(rows, model, reference_time)
    return current.forecast(rows, model, reference_time)


def predict_vector(x, model):
    if isinstance(model, dict):
        import baseline_forecast
        return baseline_forecast.predict_vector(x, model)
    hourly, _, _ = current.predict(x, expand(x), model)
    return hourly


if __name__ == '__main__':
    current.main()
