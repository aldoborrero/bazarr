# Bazarr with Submate provider
# Multi-stage build: frontend + backend

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Production image
FROM alpine:3.21

# Install dependencies
RUN apk add --no-cache --virtual=build-dependencies \
    build-base \
    cargo \
    libffi-dev \
    libxml2-dev \
    libxslt-dev \
    python3-dev && \
  apk add --no-cache \
    ffmpeg \
    libxml2 \
    libxslt \
    mediainfo \
    python3 \
    py3-pip \
    7zip && \
  mkdir -p /app/bazarr/bin /config

WORKDIR /app/bazarr/bin

# Copy backend
COPY requirements.txt ./
COPY bazarr.py ./
COPY libs ./libs
COPY custom_libs ./custom_libs
COPY bazarr ./bazarr
COPY migrations ./migrations

# Copy built frontend
COPY --from=frontend-builder /app/build ./frontend/build

# Install Python dependencies
RUN pip install --break-system-packages -U --no-cache-dir \
    --find-links https://wheel-index.linuxserver.io/alpine-3.21/ \
    -r requirements.txt && \
  apk del build-dependencies

# Environment
ENV PYTHONPATH="/app/bazarr/bin/custom_libs:/app/bazarr/bin/libs:/app/bazarr/bin/bazarr:/app/bazarr/bin"
ENV BAZARR_VERSION="submate"

EXPOSE 6767
VOLUME /config

CMD ["python3", "bazarr.py", "--no-update", "--config", "/config"]
