import functions_framework
import gcsfs
from google.cloud import storage
from prophet import Prophet
from sklearn.impute import KNNImputer
from multiprocessing import Pool
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from datetime import datetime, timedelta

bucket_store = "prediswiss-predict-storage"
bucket_name = "prediswiss-parquet-data-daily"

@functions_framework.http
def predict(request):
    request_json = {}
    content_type = request.headers["content-type"]
    if content_type == "application/json":
        request_json = request.get_json(silent=True)

    dataset = "2023.parquet"
    speed = 'speed_12'
    id = request_json['id']
    target = 'flow_11'
    date = 'publication_date'
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    store = request_json['store']

    fs_gcs = gcsfs.GCSFileSystem()
    path = bucket_name + "/" + dataset
    dataset = pq.ParquetDataset(path, filesystem=fs_gcs, filters=[('id', '=', id)])
    state_df = dataset.read(columns=[date, target, speed]).to_pandas()

    if state_df.empty:
        return ""

    state_df = state_df.sort_values(date)
    state_df[target] = pd.to_numeric(state_df[target], errors='coerce')
    state_df[speed] = pd.to_numeric(state_df[speed], errors='coerce')
    state_df[date] = pd.to_datetime(state_df[date])

    if np.isnan(state_df[speed].max()) or np.isnan(state_df[target].max()):
        return ""

    state_df[target] = imputer.fit_transform(state_df[[target]])
    state_df[speed] = imputer.fit_transform(state_df[[speed]])
    state_df.index = state_df[date]
    state_df.drop(columns=['publication_date'], axis=1, inplace=True)

    max_speed = state_df[speed].max()

    speed_mapping = {
        120: 120,
        100: 100,
        80: 80,
    }

    for condition, value in speed_mapping.items():
        if max_speed > condition:
            max_speed = value
            break

    state_df['drive'] = False
    state_df['precedent_speed'] = state_df['speed_12'].shift(1).fillna(0)
    state_df['precedent_speed'] = state_df['precedent_speed'].replace(0, np.nan)
    state_df['precedent_speed'].fillna(method='ffill', inplace=True)
    state_df['drive'] = ((state_df['speed_12'] > 0) | ((state_df['speed_12'] == 0) & ((state_df['precedent_speed'].shift(1) > max_speed / 2) | (state_df['speed_12'].shift(1) > max_speed / 2))))
    state_df.loc[(state_df['speed_12'] == 0) & (state_df['drive'] == True), 'speed_12'] = max_speed

    state_df['y'] = state_df['speed_12']
    state_df.drop(columns=[speed, target, 'speed_12'], axis=1, inplace=True)
    state_df['ds'] = state_df.index
    state_df['ds'] = state_df['ds'].dt.tz_localize(None)

    from prophet import Prophet
    model = Prophet()
    model.fit(state_df)

    targetDate = request_json['date']
    targetHour = request_json['hour']
    last_date = state_df['ds'].max()

    targetDateTime = targetDate + " " + targetHour

    format = "%Y-%m-%d %H:%M"
    datetime_obj = datetime.strptime(targetDateTime, format)

    last_date = last_date.replace(tzinfo=datetime_obj.tzinfo)

    minutes_difference = int((datetime_obj - last_date).total_seconds() / 60)

    future = model.make_future_dataframe(periods=minutes_difference, freq='min')
    forecast = model.predict(future)

    result = forecast[:minutes_difference]

    if store: 
        if fs_gcs.exists(bucket_store) == False:
            fs_gcs.mkdir(bucket_store)
        
        with fs_gcs.open(f"{bucket_store}/{id}", 'w') as file:
            file.write(result)

        return "ok"
    else:
        return result.to_json()
