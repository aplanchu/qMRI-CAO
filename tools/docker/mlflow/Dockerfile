# Docker multistage build to reduce image size
FROM python:3.7 AS build
RUN python -m venv /opt/venv

RUN apt-get update && apt-get -y install --no-install-recommends build-essential

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install mlflow boto3 pymysql

FROM python:3.7-slim
COPY --from=build /opt/venv /opt/venv

RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"