FROM python:3.12-slim-bullseye

WORKDIR /app

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

RUN pip install --no-cache-dir pipenv && \
  pipenv install --system --clear

COPY . /app

EXPOSE 1378

ENV ENVIRONMENT production

WORKDIR /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1378", "--workers", "5", "--backlog", "1024"]