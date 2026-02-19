FROM python:3.12-slim

WORKDIR /app

# Install build deps for sqlite-vec
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY maasv/ maasv/

RUN pip install --no-cache-dir ".[server,anthropic,voyage]"

# Default data directory and non-root user
RUN mkdir -p /data && \
    useradd -r -s /bin/false maasv && \
    chown maasv:maasv /data

ENV MAASV_DB_PATH=/data/maasv.db

USER maasv
EXPOSE 18790

CMD ["uvicorn", "maasv.server.main:app", "--host", "0.0.0.0", "--port", "18790"]
