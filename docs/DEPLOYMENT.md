# AtlasClaw Enterprise Deployment Guide

> **For Enterprise Customers**: This guide provides instructions for deploying AtlasClaw using Docker Compose with MySQL 8.5.

---

## Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Disk | 50 GB SSD | 200+ GB SSD |
| Docker | 20.10+ | Latest |
| Docker Compose | 2.0+ | Latest |
| MySQL | 8.5 LTS | 8.5 LTS |

---

## Quick Start

### 1. Prepare Directory Structure

```bash
mkdir -p /opt/atlasclaw/{config,data,logs,backups}
cd /opt/atlasclaw
```

### 2. Create Configuration File

Create `/opt/atlasclaw/config/atlasclaw.json`:

```json
{
  "workspace": {
    "path": "./data"
  },
  "database": {
    "type": "mysql",
    "mysql": {
      "host": "mysql",
      "port": 3306,
      "database": "atlasclaw",
      "user": "atlasclaw",
      "password": "change-to-secure-password",
      "charset": "utf8mb4"
    },
    "pool_size": 20,
    "max_overflow": 30
  },
  "model": {
    "primary": "deepseek-main",
    "fallbacks": [],
    "temperature": 0.2,
    "selection_strategy": "health",
    "tokens": [
      {
        "id": "deepseek-main",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "your-api-key-here",
        "api_type": "openai",
        "priority": 100,
        "weight": 100
      }
    ]
  },
  "service_providers": {},
  "auth": {
    "provider": "api_key",
    "api_key": {
      "keys": {
        "sk-production-key": {
          "user_id": "admin",
          "roles": ["admin"]
        }
      }
    }
  }
}
```

### 3. Create Docker Compose File

Create `/opt/atlasclaw/docker-compose.yml`:

```yaml
version: '3.8'

services:
  atlasclaw:
    image: atlasclaw-core:latest
    container_name: atlasclaw
    ports:
      - "8000:8000"
    volumes:
      - ./config/atlasclaw.json:/app/atlasclaw.json:ro
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      mysql:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  mysql:
    image: mysql:8.5
    container_name: atlasclaw-mysql
    environment:
      MYSQL_ROOT_PASSWORD: change-to-secure-root-password
      MYSQL_DATABASE: atlasclaw
      MYSQL_USER: atlasclaw
      MYSQL_PASSWORD: change-to-secure-password
    volumes:
      - ./mysql-data:/var/lib/mysql
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "root", "-pchange-to-secure-root-password"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 60s
    command:
      - --character-set-server=utf8mb4
      - --collation-server=utf8mb4_unicode_ci
      - --default-authentication-plugin=mysql_native_password
```

### 4. Start Services

```bash
cd /opt/atlasclaw
docker-compose up -d
```

### 5. Run Database Migrations

```bash
docker-compose exec atlasclaw alembic upgrade head
```

### 6. Verify Deployment

```bash
# Check container status
docker-compose ps

# Health check
curl http://localhost:8000/api/health
# Expected: {"status": "healthy", "timestamp": "..."}
```

---

## Configuration Reference

### Database

```json
{
  "database": {
    "type": "mysql",
    "mysql": {
      "host": "mysql",
      "port": 3306,
      "database": "atlasclaw",
      "user": "atlasclaw",
      "password": "secure-password",
      "charset": "utf8mb4"
    },
    "pool_size": 20,
    "max_overflow": 30
  }
}
```

### LLM Provider

```json
{
  "model": {
    "primary": "deepseek-main",
    "fallbacks": [],
    "temperature": 0.2,
    "tokens": [
      {
        "id": "deepseek-main",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "your-api-key",
        "api_type": "openai"
      }
    ]
  }
}
```

### Authentication

**API Key:**
```json
{
  "auth": {
    "provider": "api_key",
    "api_key": {
      "keys": {
        "sk-your-key": {
          "user_id": "admin",
          "roles": ["admin"]
        }
      }
    }
  }
}
```

**OIDC/OAuth2:**
```json
{
  "auth": {
    "provider": "oidc",
    "oidc": {
      "issuer": "https://auth.company.com",
      "client_id": "atlasclaw-client",
      "client_secret": "client-secret",
      "redirect_uri": "https://atlasclaw.company.com/api/auth/callback"
    }
  }
}
```

---

## Operations

### View Logs

```bash
docker-compose logs -f atlasclaw
docker-compose logs -f mysql
```

### Backup

```bash
#!/bin/bash
# /opt/atlasclaw/backup.sh

DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker exec atlasclaw-mysql mysqldump -u root -p'root-password' atlasclaw | gzip > ./backups/atlasclaw_${DATE}.sql.gz

# Backup config and data
tar -czf ./backups/atlasclaw_data_${DATE}.tar.gz ./config ./data

# Keep only last 30 days
find ./backups -name "*.gz" -mtime +30 -delete
```

### Update

```bash
# Pull latest image
docker-compose pull atlasclaw

# Restart with migrations
docker-compose up -d
docker-compose exec atlasclaw alembic upgrade head
```

### Stop

```bash
docker-compose down
```

---

## Troubleshooting

### Service Not Starting

```bash
# Check logs
docker-compose logs atlasclaw

# Check config syntax
docker-compose exec atlasclaw python -c "import json; json.load(open('atlasclaw.json'))"
```

### Database Connection Failed

```bash
# Check MySQL is healthy
docker-compose ps mysql

# Test connection manually
docker-compose exec mysql mysql -u atlasclaw -p -e "SELECT 1"
```

---

## Security Notes

1. Change all default passwords in `atlasclaw.json` and `docker-compose.yml`
2. Restrict file permissions: `chmod 600 config/atlasclaw.json`
3. Use HTTPS in production (place a reverse proxy in front)
4. Regularly backup data directory and database

---

For detailed configuration options, refer to `atlasclaw.json.example` in the source repository.
