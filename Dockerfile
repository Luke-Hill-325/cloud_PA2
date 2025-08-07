FROM apache/spark:python3

WORKDIR /opt/spark/work-dir

COPY Predictions.py /opt/spark/work-dir
COPY ValidationDataset.csv /opt/spark/work-dir

RUN pip install --trusted-host pypi.python.org --no-cache-dir --target="packages" pandas numpy findspark

CMD ["python3", "Predictions.py"]
