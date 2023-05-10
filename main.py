import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import pyarrow as pa
from datetime import datetime, timedelta
from dateutil import rrule
import gcsfs
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

bucket_name = "prediswiss-parquet-data"

def main():
    datasetPath = "2023-05"
    fs_gcs = gcsfs.GCSFileSystem()
    path = bucket_name + "/" + datasetPath + ".parquet"
    table = pq.read_table(path, filesystem=fs_gcs)
    df = table.to_pandas()


    #print(df.groupby(["publication_date"]).count())
    with open('test.txt', 'w') as f:
        f.write(df.groupby(["publication_date"]).count().to_string())

    
    flow_df = df[df['id'] == 'CH:0542.05']
    flow_df = flow_df[['flow_11']]
    flow_df = flow_df.dropna().values.astype(int)
    print(flow_df)

    # Fit an ARIMA model to the specified column
    model = ARIMA(flow_df, order=(1,1,1))
    
    model_fit = model.fit()
    
    # Make predictions for the specified number of periods into the future
    forecast = model_fit.forecast(10)

    print(forecast)

if __name__ == "__main__":
    main()