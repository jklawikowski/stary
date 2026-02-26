ARG BASE_IMAGE=ger-is-registry-pre.caas.intel.com/qa-platform/base/ubuntu-intel:22.04@sha256:8353ed6c7075572269986e8c60c9adae6fed80fcb9fc392d03eb534de058e0a2
FROM $BASE_IMAGE

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Corporate proxy
ENV HTTP_PROXY=http://proxy-dmz.intel.com:912/
ENV http_proxy=http://proxy-dmz.intel.com:912/
ENV HTTPS_PROXY=http://proxy-dmz.intel.com:912/
ENV https_proxy=http://proxy-dmz.intel.com:912/
ENV NO_PROXY=0.0.0.0,127.0.0.1,localhost,intel.com,habana-labs.com
ENV no_proxy=0.0.0.0,127.0.0.1,localhost,intel.com,habana-labs.com

RUN apt-get update && apt-get install -y \
    gcc \
    git \
    python3 \
    python3-pip \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

COPY requirements/dagster.txt /requirements/dagster.txt
RUN pip3 install -r /requirements/dagster.txt

ENV DAGSTER_HOME=/opt/dagster/dagster_home/
RUN mkdir -p $DAGSTER_HOME
COPY dagster/dev/dagster/dagster.yaml dagster/dev/dagster/workspace.yaml $DAGSTER_HOME
WORKDIR $DAGSTER_HOME

EXPOSE 3000

CMD ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000"]
