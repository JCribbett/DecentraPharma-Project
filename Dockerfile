# Dockerfile

# --- Base Image ---
FROM python:3.10-slim-buster

# --- Environment Variables ---
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# --- Working Directory ---
WORKDIR /app

# --- Install System Dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Project Files ---
COPY . .

# --- IPFS Daemon (Optional) ---
# Uncomment below to install and run IPFS within the container:
# RUN apt-get update && apt-get install -y wget \
#     && wget https://dist.ipfs.io/kubo/v0.25.0/kubo_v0.25.0_linux-amd64.tar.gz \
#     && tar xvf kubo_v0.25.0_linux-amd64.tar.gz \
#     && mv kubo /usr/local/bin/ipfs \
#     && ipfs init \
#     && rm kubo_v0.25.0_linux-amd64.tar.gz
# EXPOSE 5001
# EXPOSE 8080

# --- Expose Ports ---
# Ray dashboard (default)
EXPOSE 8265
# Ray client (default)
EXPOSE 10001

# --- Application Command ---
# Development placeholder — keeps container running for interactive use.
# Replace with actual entry point when ready, e.g.:
# CMD ["python", "src/cli/main.py"]
CMD ["tail", "-f", "/dev/null"]
