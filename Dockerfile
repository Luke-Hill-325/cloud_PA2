from apache/spark:4.0.0-python3

workdir /opt/app

copy Predictions.py /opt/app
copy ValidationDataset.csv /opt/app

run pip install --no-cache-dir pandas numpy findspark

cmd ["python", "Predictions.py"]
