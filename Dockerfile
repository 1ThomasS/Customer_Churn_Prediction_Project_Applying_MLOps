FROM python:3.10.9

COPY requirements.txt /app/

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app