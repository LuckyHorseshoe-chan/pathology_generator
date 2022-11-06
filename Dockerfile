FROM ubuntu:20.04
FROM python:3.7
RUN pip install --upgrade pip
RUN pip install pydicom keras psycopg2 tensorflow
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN pip install "fastapi[all]"
COPY ./ /app
CMD uvicorn main:app --reload
