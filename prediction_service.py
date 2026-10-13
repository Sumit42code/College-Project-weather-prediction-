"""Shared, notebook-aligned inference used locally and in SageMaker."""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURES = ['forecast_precipitation_mm', 'forecast_max_wind_kmh']
MAX_ROWS = 500


def load_package(path):
    # Load only your own exported model files, never user uploads.
    package = joblib.load(Path(path))
    if not isinstance(package, dict):
        raise ValueError('Model file must be the exported model_package dictionary.')
    for key in ['model', 'model_name', 'threshold', 'features']:
        if key not in package:
            raise ValueError(f'Model package is missing {key}. Re-export it from the notebook.')
    if list(package['features']) != FEATURES:
        raise ValueError('Expected the two forecast features from the supplied notebook.')
    value = float(package['threshold'])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Model warning threshold must be between 0 and 1.')
    classes = list(package['model'].classes_)
    if 1 not in classes:
        raise ValueError('The fitted model must include unsuitable class 1.')
    return package


def validate_records(records):
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_ROWS:
        raise ValueError(f'Supply 1 to {MAX_ROWS} daily forecast records.')
    rows = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'Row {i + 1} must be an object.')
        row = {}
        for feature in FEATURES:
            raw = record.get(feature)
            if raw is None or isinstance(raw, (bool, str)):
                raise ValueError(f'Row {i + 1}: {feature} must be a number.')
            try:
                number = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f'Row {i + 1}: invalid {feature}.') from None
            if not math.isfinite(number) or number < 0:
                raise ValueError(f'Row {i + 1}: {feature} must be finite and non-negative.')
            row[feature] = number
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)


def package_metadata(package):
    limits = package.get('weather_limits', {'precipitation_mm': 2.0, 'wind_kmh': 30.0})
    for key in ['precipitation_mm', 'wind_kmh']:
        value = float(limits[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid exported weather limits.')
    return {
        'schema_version': 1,
        'model_name': str(package['model_name']),
        'model_version': str(package.get('model_version', 'legacy-export')),
        'threshold': float(package['threshold']),
        'weather_limits': {key: float(limits[key]) for key in limits},
    }


def predict_local(package, records):
    frame = validate_records(records)
    metadata = package_metadata(package)
    classes = list(package['model'].classes_)
    probabilities = np.asarray(package['model'].predict_proba(frame))[:, classes.index(1)]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Model returned invalid probabilities.')
    metadata['predictions'] = [
        {'unsuitable_probability': float(p), 'unsuitable': int(p >= metadata['threshold'])}
        for p in probabilities
    ]
    return metadata


def validate_response(response, expected_rows):
    if not isinstance(response, dict) or response.get('schema_version') != 1:
        raise ValueError('AWS endpoint returned an incompatible response. Deploy the included serving code.')
    for key in ['model_name', 'model_version', 'threshold', 'weather_limits', 'predictions']:
        if key not in response:
            raise ValueError(f'Endpoint response is missing {key}.')
    threshold = response['threshold']
    if isinstance(threshold, (bool, str)) or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError('Endpoint returned an invalid warning threshold.')
    limits = response['weather_limits']
    if not isinstance(limits, dict):
        raise ValueError('Endpoint returned invalid weather limits.')
    for key in ['precipitation_mm', 'wind_kmh']:
        value = limits.get(key)
        if isinstance(value, (bool, str)) or value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError('Endpoint returned invalid weather limits.')
    predictions = response['predictions']
    if not isinstance(predictions, list) or len(predictions) != expected_rows:
        raise ValueError('Endpoint returned the wrong number of predictions.')
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError('Invalid endpoint prediction.')
        p = item.get('unsuitable_probability')
        if p is None or isinstance(p, (bool, str)) or not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
            raise ValueError('Endpoint returned an invalid probability.')
        if item.get('unsuitable') not in [0, 1] or item['unsuitable'] != int(float(p) >= float(threshold)):
            raise ValueError('Endpoint prediction disagrees with its exported threshold.')
    return response


def predict_aws(records, endpoint_name, region, runtime=None):
    validate_records(records)
    if not endpoint_name or not region:
        raise ValueError('Configure WEATHER_ENDPOINT_NAME and AWS_DEFAULT_REGION first.')
    if runtime is None:
        import boto3
        from botocore.config import Config
        runtime = boto3.client('sagemaker-runtime', region_name=region,
            config=Config(connect_timeout=10, read_timeout=70,
                          retries={'mode': 'standard', 'max_attempts': 2}))
    result = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps({'instances': records}, allow_nan=False).encode('utf-8'),
    )
    return validate_response(json.loads(result['Body'].read().decode('utf-8')), len(records))
