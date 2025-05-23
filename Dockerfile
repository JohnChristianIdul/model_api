FROM tiangolo/uvicorn-gunicorn-fastapi-docker:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app