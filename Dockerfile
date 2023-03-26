FROM python:3.9-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py
