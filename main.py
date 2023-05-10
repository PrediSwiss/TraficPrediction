import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import pyarrow as pa
from datetime import datetime, timedelta
from dateutil import rrule
import gcsfs
from google.cloud import storage

bucket_name = "prediswiss-parquet-data"

def main():
    datasetPath = "2023-05"
    fs_gcs = gcsfs.GCSFileSystem()
    path = bucket_name + "/" + datasetPath + ".parquet"
    table = pq.read_table(path, filesystem=fs_gcs)
    df = table.to_pandas()
    print(df.groupby(["publication_date"]).count())
    with open('test.txt', 'w') as f:
        f.write(df.groupby(["publication_date"]).count().to_string())

if __name__ == "__main__":
    main()