"""Run: python -m streamlit run Market_Weather_Decision_app.py"""
from pathlib import Path
import io
import json
import math
import os
import logging

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ['forecast_precipitation_mm', 'forecast_max_wind_kmh']
MAX_ROWS = 500


def load_package(path):
    # Load your own notebook export, never a model uploaded by an app visitor.
    value = joblib.load(Path(path))
    if isinstance(value, dict):
        package = dict(value)
        for key in ['model', 'model_name', 'threshold', 'features']:
            if key not in package:
                raise ValueError(f'Model package is missing {key}. Use Export_Weather_Model.py.')
    else:
        # A plain joblib.dump(final_model, ...) file also runs. It cannot retain
        # the notebook's selected threshold or evidence without metadata.
        metadata_path = Path(path).with_name('model_metadata.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
        package = {'model': value, 'model_name': type(value).__name__,
                   'threshold': 0.5, 'features': FEATURES,
                   'weather_limits': {'precipitation_mm': 2.0, 'wind_kmh': 30.0},
                   'model_version': 'plain-model', **metadata}
        package['_plain_model'] = True
        package['_defaults_used'] = not metadata_path.exists()
    if list(package['features']) != FEATURES:
        raise ValueError('Expected forecast_precipitation_mm, then forecast_max_wind_kmh.')
    model = package['model']
    if not hasattr(model, 'predict_proba') or not hasattr(model, 'classes_'):
        raise ValueError('Export a fitted classifier with predict_proba and classes_.')
    classes = list(model.classes_)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError('Model classes must be suitable=0 and unsuitable=1.')
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None:
        candidate = getattr(model, 'feature_names_', None)
        if candidate is not None and len(candidate) and list(candidate) != [str(i) for i in range(len(candidate))]:
            feature_names = candidate
    if feature_names is not None and list(feature_names) != FEATURES:
        raise ValueError('The model feature names/order do not match this app.')
    # Some CatBoost versions report n_features_in_=0 after joblib reload.
    count = getattr(model, 'n_features_in_', None)
    if count not in (None, 0, len(FEATURES)):
        raise ValueError('The model must be trained on exactly two forecast features.')
    value = float(package['threshold'])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Model warning threshold must be between 0 and 1.')
    package_metadata(package)  # Validate both exported weather limits.
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
        raise ValueError('AWS endpoint returned an incompatible response. Use the project SageMaker serving contract.')
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


ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get('WEATHER_ARTIFACTS_DIR', str(ROOT / 'artifacts')))
model_candidates = ([ARTIFACTS / 'final_weather_model.joblib'] if os.environ.get('WEATHER_ARTIFACTS_DIR')
                    else [ROOT / 'final_weather_model.joblib', ARTIFACTS / 'final_weather_model.joblib'])
MODEL_PATH = next((path for path in model_candidates if path.exists()), model_candidates[0])
LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title='Market Weather Decision Support', page_icon='🌤️', layout='wide')
st.markdown("""<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0e1117; color: #fafafa;
}
[data-testid="stHeader"] {background: #0e1117;}
.block-container {max-width: 1050px; padding-top: 4rem; padding-bottom: 3rem;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {color: #fafafa;}
.stApp h1 {font-size: 2.5rem; letter-spacing: -0.035em;}
[data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p,
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] {color: #fafafa;}
[data-testid="stCaptionContainer"] p {color: #a0a3ad;}
[data-testid="stForm"] {border: 0; padding: 0; background: transparent;}
[data-baseweb="input"], [data-baseweb="input"] > div,
[data-testid="stNumberInput"] input {
    background: #262730 !important; color: #fafafa !important; border-color: #31333f !important;
}
[data-testid="stNumberInput"] button {background: #262730; color: #fafafa;}
button[kind="primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    background: #ff4b4b !important; border-color: #ff4b4b !important; color: #ffffff !important;
    min-height: 2.65rem;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
    background: #171a22; color: #fafafa; border-color: #343741;
}
[data-testid="stExpander"] {border-color: #343741; color: #fafafa;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {background: #0e1117; color: #fafafa;}
[data-testid="stFileUploaderDropzone"] {background: #171a22; color: #fafafa;}
.stApp hr {border-color: #343741;}
.recommendation {padding: 1rem; border-radius: 0.5rem; margin: 0.25rem 0 1rem; font-weight: 600;}
.recommendation.good {background: #153c29; color: #36d17b;}
.recommendation.caution {background: #48351a; color: #ffc857;}
.risk-track {height: 9px; width: 100%; background: #262730; border-radius: 6px; overflow: hidden; margin: 0.5rem 0;}
.risk-fill {height: 100%; background: #1c83e1; border-radius: 6px;}
.weather-table-wrap {border: 1px solid #343741; border-radius: 8px; overflow: auto; margin: 0.5rem 0 1rem; max-height: 390px;}
.weather-table {width: 100%; border-collapse: collapse; color: #fafafa; font-size: 0.95rem;}
.weather-table th {background: #1c202b; text-align: left; padding: 0.8rem; font-weight: 600; white-space: nowrap;}
.weather-table td {padding: 0.8rem; border-top: 1px solid #343741; vertical-align: top;}
.weather-table tr:nth-child(even) td {background: #131720;}
</style>""", unsafe_allow_html=True)
st.title('🌤️ Market Weather Decision Support')
st.write('Assess whether forecast weather conditions are suitable for an upcoming market day.')


def setting(key, default=''):
    value = os.environ.get(key)
    if value is not None:
        return value
    try:
        return str(st.secrets.get(key, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


@st.cache_resource
def cached_package(path, stamp):
    return load_package(path)


package = None
load_failed = False
if MODEL_PATH.exists():
    try:
        package = cached_package(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except Exception:
        load_failed = True
        LOGGER.exception('Could not load the weather model')
endpoint = setting('WEATHER_ENDPOINT_NAME')
region = setting('AWS_DEFAULT_REGION', 'ap-southeast-2')
method = 'AWS' if endpoint else ('Local' if package is not None or load_failed else 'Basic')
service_unavailable = load_failed and not endpoint
if service_unavailable:
    st.error('The weather assessment is temporarily unavailable. Please try again later.')
elif method == 'Basic':
    st.info('A basic rain and wind check is available. The trained assessment and probability are not available today.')


def forecast_rule(records, limits):
    data = validate_records(records)
    return ((data[FEATURES[0]] > limits['precipitation_mm'])
            | (data[FEATURES[1]] > limits['wind_kmh'])).astype(int)


def assess(records):
    if method == 'AWS':
        return predict_aws(records, endpoint, region)
    if method == 'Local':
        if package is None:
            raise RuntimeError('Weather model unavailable')
        return predict_local(package, records)
    limits = {'precipitation_mm': 2.0, 'wind_kmh': 30.0}
    return {'threshold': None, 'weather_limits': limits,
            'predictions': [{'unsuitable': int(value), 'unsuitable_probability': None}
                            for value in forecast_rule(records, limits)]}


def recommendation(warning):
    return 'Prepare a backup plan' if warning else 'Suitable to proceed'


def display_table(frame):
    html = frame.to_html(index=False, border=0, classes='weather-table', escape=True)
    st.markdown('<div class="weather-table-wrap">' + html + '</div>', unsafe_allow_html=True)


def expected_error_cost(probability, warning, missed_cost, false_alarm_cost):
    """Illustrative expected error cost, using one shared probability estimate.

    No warning: P(unsuitable) * loss from a missed unsuitable day.
    Warning: P(suitable) * cost of an unnecessary backup.
    Correct decisions and normal operating costs are excluded, matching the
    notebook's false-negative / false-positive cost convention.
    """
    values = [probability, missed_cost, false_alarm_cost]
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError('Probability and costs must be finite.')
    if not 0 <= probability <= 1 or missed_cost < 0 or false_alarm_cost < 0:
        raise ValueError('Invalid probability or cost assumption.')
    return float((1 - probability) * false_alarm_cost if warning else probability * missed_cost)


PUBLIC_COLUMNS = ['Market date', 'Rainfall (mm)', 'Wind speed (km/h)']


def prepare_upload(data):
    """Accept friendly template headings and earlier project CSV headings."""
    if data.empty or len(data) > MAX_ROWS:
        raise ValueError(f'Please upload between 1 and {MAX_ROWS} market dates.')
    aliases = {'Market date': 'market_date', 'Rainfall (mm)': FEATURES[0],
               'Wind speed (km/h)': FEATURES[1]}
    for label, internal in aliases.items():
        if label in data.columns and internal in data.columns:
            raise ValueError('Your file has repeated weather columns. Please use the upload template.')
    frame = data.rename(columns=aliases).copy()
    if not set(FEATURES).issubset(frame.columns):
        raise ValueError('Please include Rainfall (mm) and Wind speed (km/h). You can use the template below.')
    try:
        weather = frame[FEATURES].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError):
        raise ValueError('Rainfall and wind speed must contain numbers only.') from None
    if any(frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any() for column in FEATURES):
        raise ValueError('Rainfall and wind speed must contain numbers only.')
    if not np.isfinite(weather.to_numpy(dtype=float)).all() or (weather < 0).any().any():
        raise ValueError('Please fill in every rainfall and wind value with a number of zero or more.')
    public = pd.DataFrame(index=frame.index)
    if 'market_date' in frame.columns:
        dates = frame['market_date'].fillna('').astype(str).str.strip()
        try:
            parsed = pd.to_datetime(dates, format='%Y-%m-%d', errors='raise')
        except (ValueError, TypeError):
            raise ValueError('Use YYYY-MM-DD for market dates, for example 2026-10-20.') from None
        if parsed.isna().any() or not dates.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            raise ValueError('Please fill in each market date using YYYY-MM-DD.')
        public['Market date'] = parsed.dt.strftime('%Y-%m-%d')
    else:
        public['Market date'] = [f'Date {number + 1}' for number in range(len(frame))]
    public['Rainfall (mm)'] = weather[FEATURES[0]]
    public['Wind speed (km/h)'] = weather[FEATURES[1]]
    return public.reset_index(drop=True), weather.to_dict('records')


service_context = (method, str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else None,
                   endpoint, region)
if st.session_state.get('_market_ui_version') != 3:
    # Clear results from the earlier, simpler interface after a hot reload.
    st.session_state.pop('market_assessment', None)
    st.session_state.pop('batch_assessment', None)
    st.session_state['_market_ui_version'] = 3

st.subheader('Forecast Conditions')
st.caption('Palmerston North • 10 am–4 pm • Use forecasts made about seven days ahead')
with st.form('weather_decision'):
    left, right = st.columns(2)
    rain = left.number_input('Forecast precipitation (mm)', min_value=0.0, value=1.0, step=0.1,
                             format='%.2f', key='rainfall',
                             help='Total rain, showers and snow expected during the market hours. Replace the example value with your forecast.')
    wind = right.number_input('Forecast maximum wind (km/h)', min_value=0.0, value=20.0, step=1.0,
                              format='%.2f', key='wind_speed',
                              help='Highest wind speed expected during the market hours. Replace the example value with your forecast.')
    submit = st.form_submit_button('Assess Market Day', type='primary', width='stretch', disabled=service_unavailable)
if submit:
    st.session_state.pop('market_assessment', None)
    record = {FEATURES[0]: rain, FEATURES[1]: wind}
    try:
        with st.spinner('Assessing the forecast…'):
            response = assess([record])
        row = response['predictions'][0]
        limits = response['weather_limits']
        st.session_state['market_assessment'] = {
            'context': service_context, 'rain': rain, 'wind': wind,
            'warning': bool(row['unsuitable']), 'probability': row['unsuitable_probability'],
            'threshold': response['threshold'], 'limits': limits,
            'rule_warning': bool(forecast_rule([record], limits).iloc[0])}
    except Exception:
        LOGGER.exception('Market weather check failed')
        st.error('We couldn’t assess this forecast right now. Please try again later.')

saved = st.session_state.get('market_assessment')
if saved and saved['context'] == service_context:
    warning = saved['warning']
    p = saved['probability']
    limits = saved['limits']
    rule_warning = saved['rule_warning']
    st.divider()
    st.subheader('Market Day Recommendation')
    if warning:
        st.markdown('<div class="recommendation caution">⚠️ Prepare a backup plan</div>', unsafe_allow_html=True)
        st.write('The forecast suggests weather could disrupt your market. Check indoor or sheltered arrangements.')
    else:
        st.markdown('<div class="recommendation good">✅ Suitable to proceed</div>', unsafe_allow_html=True)
        st.write('The forecast conditions are assessed as suitable for proceeding with the market.')
    if p is not None:
        left, right = st.columns(2)
        left.metric('Probability of unsuitable weather', f'{p:.2%}')
        right.metric('Decision threshold', f"{saved['threshold']:.2%}")
        comparison = 'at or above' if warning else 'below'
        st.write(f"The predicted probability of unsuitable weather is **{p:.2%}**, which is {comparison} "
                 f"the **{saved['threshold']:.2%}** decision threshold.")
        st.subheader('Unsuitable Weather Risk')
        st.markdown(f'<div class="risk-track" role="progressbar" aria-label="Estimated unsuitable weather risk" '
                    f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{p * 100:.2f}">'
                    f'<div class="risk-fill" style="width:{p * 100:.3f}%"></div></div>', unsafe_allow_html=True)
        st.caption('0% = low predicted risk | 100% = high predicted risk')
    else:
        st.caption('This basic check does not provide a probability estimate.')
    with st.expander('About the prediction'):
        st.write(f"This assessment uses {saved['rain']:g} mm of forecast precipitation and "
                 f"{saved['wind']:g} km/h maximum wind for the market hours, 10 am–4 pm.")
        st.write(f"The project treats weather as unsuitable when precipitation exceeds {limits['precipitation_mm']:g} mm "
                 f"or wind exceeds {limits['wind_kmh']:g} km/h. These are planning assumptions.")
        st.write('Use the recommendation to plan ahead, then review updated forecasts before the market. '
                 'It is not an automatic cancellation decision.')
    st.divider()
    st.subheader('Baseline Comparison')
    st.write('Compare the trained recommendation with a simple rain-and-wind check using the same forecast.')
    comparison_rows = [{'Approach': 'Trained model', 'Recommendation': recommendation(warning)},
                       {'Approach': 'Simple weather rule (baseline)', 'Recommendation': recommendation(rule_warning)}]
    if method == 'Basic':
        comparison_rows = comparison_rows[1:]
        st.caption('The trained comparison is unavailable; the simple weather rule is shown below.')
    display_table(pd.DataFrame(comparison_rows))
    st.caption(f"The baseline warns above {limits['precipitation_mm']:g} mm of precipitation or "
               f"{limits['wind_kmh']:g} km/h wind. Equality at a weather limit does not trigger a warning.")
    if method != 'Basic' and warning != rule_warning:
        st.info('The two approaches disagree. Review the forecast and backup options before deciding.')
    st.subheader('Estimated Planning Costs')
    st.write('Estimate the cost if a planning decision turns out to be wrong.')
    with st.expander('Adjust the NZ$ cost assumptions', expanded=True):
        left, right = st.columns(2)
        missed_cost = left.number_input('Disruption without a backup (NZ$)', min_value=0.0,
                                       value=1500.0, step=100.0, format='%.2f', key='missed_cost',
                                       help='Assumed loss when unsuitable weather was missed and no backup was prepared.')
        alarm_cost = right.number_input('Unnecessary backup arrangements (NZ$)', min_value=0.0,
                                       value=500.0, step=50.0, format='%.2f', key='false_alarm_cost',
                                       help='Assumed cost when a backup was prepared but the weather turned out suitable.')
    download = {'Precipitation (mm)': saved['rain'], 'Maximum wind (km/h)': saved['wind'],
                'Recommendation': recommendation(warning), 'Baseline recommendation': recommendation(rule_warning)}
    if p is not None:
        model_cost = expected_error_cost(p, warning, missed_cost, alarm_cost)
        baseline_cost = expected_error_cost(p, rule_warning, missed_cost, alarm_cost)
        difference = baseline_cost - model_cost
        left, middle, right = st.columns(3)
        left.metric('Model recommendation', f'NZ${model_cost:,.2f}')
        middle.metric('Baseline recommendation', f'NZ${baseline_cost:,.2f}')
        right.metric('Estimated cost avoided' if difference >= 0 else 'Extra estimated cost', f'NZ${abs(difference):,.2f}')
        if warning == rule_warning:
            st.caption('Both approaches recommend the same action, so their estimated error costs are equal.')
        elif difference > 0:
            st.caption(f'The model recommendation has NZ${difference:,.2f} less estimated error cost under these assumptions.')
        elif difference < 0:
            st.caption(f'The model recommendation has NZ${-difference:,.2f} more estimated error cost under these assumptions.')
        download.update({'Unsuitable weather probability (%)': p * 100,
                         'Decision threshold (%)': saved['threshold'] * 100,
                         'Assumed disruption loss (NZ$)': missed_cost,
                         'Assumed unnecessary backup cost (NZ$)': alarm_cost,
                         'Model estimated error cost (NZ$)': model_cost,
                         'Baseline estimated error cost (NZ$)': baseline_cost})
        with st.expander('How are these costs estimated?'):
            st.write('Without a backup: chance of unsuitable weather × disruption loss. '
                     'With a backup warning: chance of suitable weather × unnecessary backup cost.')
            st.write('Both approaches use the same trained probability estimate. The baseline does not '
                     'produce its own probability. Changing costs does not change either recommendation.')
        st.caption('Illustrative planning-error estimates, not actual bills or guaranteed savings. '
                   'Their reliability depends on the probability estimate. Normal operating costs and costs '
                   'for correct decisions are excluded.')
    else:
        st.info('A trained probability is needed to estimate these costs. No cost estimates have been substituted.')
    st.download_button('Download this assessment', pd.DataFrame([download]).to_csv(index=False),
                       'market_assessment.csv', 'text/csv', key='download_market_plan')

st.divider()
with st.expander('Batch Predictions — check several market dates'):
    st.write('Upload one row per market date with the total rainfall and highest wind speed expected between 10 am and 4 pm.')
    st.caption('Use forecasts made about a week ahead. Maximum 500 rows. Dates use YYYY-MM-DD, for example 2026-10-20.')
    template = pd.DataFrame(columns=PUBLIC_COLUMNS)
    st.download_button('Download CSV template', template.to_csv(index=False),
                       'market_dates_template.csv', 'text/csv', key='download_template')
    upload = st.file_uploader('Upload your forecast CSV', type=['csv'], key='batch_upload')
    batch_context = (service_context, upload.getvalue() if upload is not None else None)
    if upload is not None and st.button('Assess uploaded dates', type='primary', disabled=service_unavailable):
        st.session_state.pop('batch_assessment', None)
        try:
            data = pd.read_csv(io.BytesIO(upload.getvalue()), nrows=MAX_ROWS + 1)
            public, records = prepare_upload(data)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            st.error('We couldn’t read that file. Please use the CSV template and try again.')
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error('We couldn’t read that file. Please use the CSV template and try again.')
        else:
            try:
                with st.spinner('Assessing your market dates…'):
                    response = assess(records)
                rows = response['predictions']
                warnings = [bool(row['unsuitable']) for row in rows]
                rules = forecast_rule(records, response['weather_limits']).astype(bool).tolist()
                public['Recommendation'] = [recommendation(value) for value in warnings]
                if method != 'Basic':
                    public['Baseline recommendation'] = [recommendation(value) for value in rules]
                    public['Unsuitable weather probability (%)'] = [round(row['unsuitable_probability'] * 100, 2) for row in rows]
                st.session_state['batch_assessment'] = {
                    'context': batch_context, 'table': public,
                    'backup_count': sum(warnings), 'outdoor_count': len(warnings) - sum(warnings)}
            except Exception:
                LOGGER.exception('Batch weather check failed')
                st.error('We couldn’t assess these dates right now. Please try again later.')
    saved_batch = st.session_state.get('batch_assessment')
    if saved_batch and saved_batch['context'] == batch_context:
        left, right = st.columns(2)
        left.metric('Dates needing a backup', saved_batch['backup_count'])
        right.metric('Dates suitable to proceed', saved_batch['outdoor_count'])
        display_table(saved_batch['table'])
        st.download_button('Download batch results', saved_batch['table'].to_csv(index=False),
                           'market_predictions.csv', 'text/csv', key='download_batch_plans')
