import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import pyarrow as pa
from datetime import datetime, timedelta
from dateutil import rrule
import gcsfs
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

bucket_name = "prediswiss-parquet-data"

class Arima:
    def __init__(self, bucket_name, datasets):
        fs_gcs = gcsfs.GCSFileSystem()
        dataframes = []
        for dataset in datasets:
            path = bucket_name + "/" + dataset + ".parquet"
            table = pq.read_table(path, filesystem=fs_gcs)
            df = table.to_pandas()
            dataframes.append(df)
        self.df = pd.concat(dataframes)

    def train_model(self, id, target, order):
        self.target = target
        self.id = id
        self.order = order
        flow_df = self.df[self.df['id'] == id]
        flow_df = flow_df[target]
        flow_df = flow_df.fillna(value=0, inplace=True)
        model = ARIMA(flow_df, order=order)
        self.model_fit = model.fit()

    def forecast(self, time):
        forecast = self.model_fit.forecast(time)
        print(forecast)

def main():
    arima = Arima(bucket_name, ["2023-05"])
    arima.train_model('CH:0542.05', 'flow_1', (1,1,3))
    arima.forecast(10)

if __name__ == "__main__":
    main()