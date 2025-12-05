FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# スクリプトを scripts/ に配置（親ディレクトリからの相対パス維持のため）
RUN mkdir -p /app/scripts /app/data/osm /app/data/sensors
COPY *.py /app/scripts/

WORKDIR /app/scripts
ENTRYPOINT ["python"]
