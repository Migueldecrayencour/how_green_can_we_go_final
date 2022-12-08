FROM python:3.10.6-buster
COPY app /app
COPY my_model /my_model
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn app.api:app --host 0.0.0.0 --port $PORT
