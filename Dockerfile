FROM python:3.9-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python -m flask run --host=0.0.0.0
