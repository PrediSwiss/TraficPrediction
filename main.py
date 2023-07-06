import functions_framework
import gcsfs
from google.cloud import storage
from prophet import Prophet
from sklearn.impute import KNNImputer
from multiprocessing import Pool
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt

# {"id": 'CH:123', "time": 3600, "store": True}
@functions_framework.http
def predict(request):
    dataset = "2023.parquet"
    bucket_name = "prediswiss-parquet-data-daily"

    fs_gcs = gcsfs.GCSFileSystem()
    path = bucket_name + "/" + dataset + ".parquet"
    table = pq.read_table(path, filesystem=fs_gcs)
    df = table.to_pandas()

    speed = 'speed_12'
    id = request['id']
    target = 'flow_11'
    date = 'publication_date'
    imputer = KNNImputer(n_neighbors=2, weights="uniform")

    state_df = df[df['id'] == id].copy()

    state_df = state_df[[date, target, speed]]
    state_df = state_df.sort_values(date)
    state_df[target] = pd.to_numeric(state_df[target], errors='coerce')
    state_df[speed] = pd.to_numeric(state_df[speed], errors='coerce')
    state_df[date] = pd.to_datetime(state_df[date])
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

    future = model.make_future_dataframe(periods=request['time'], freq='min')
    forecast = model.predict(future)

    return forecast[:request['time']].to_json()