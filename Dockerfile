

FROM python:3.7
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./model_1.pkl /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]