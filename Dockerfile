from apache/spark:4.0.0-python3

workdir /opt/app

copy Training.py /opt/app
copy TrainingDataset.csv /opt/app
copy ValidationDataset.csv /opt/app

run pip install --no-cache-dir pandas numpy
