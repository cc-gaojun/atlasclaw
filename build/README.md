# AtlasClaw Deployment Guide

This guide describes how to deploy AtlasClaw in your own environment using pre-built Docker images.

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk | 20 GB SSD | 100+ GB SSD |
| OS | Linux (CentOS 7+, Ubuntu 18.04+, or equivalent) | Latest LTS |

### Required Software

- **Docker** 20.10 or higher
- **Docker Compose** 2.0 or higher

### Install Docker

**CentOS/RHEL:**
```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

**Verify Installation:**
```bash
docker --version
docker compose version
```

---

## Quick Start

### 1. Create Deployment Directory

```bash
mkdir -p /opt/atlasclaw/{config,data,logs}
cd /opt/atlasclaw
```

**Directory Structure:**

```
/opt/atlasclaw/
├── docker-compose.yml      # Docker Compose orchestration file
├── config/
│   └── atlasclaw.json      # Configuration file (mounted to /app/atlasclaw.json in container)
├── data/                   # Database and runtime data (persisted volume)
└── logs/                   # Application logs (persisted volume)
```

The `config/atlasclaw.json` is mounted to `/app/atlasclaw.json` inside the container where the application reads its configuration.

### 2. Download Compose File

Download the appropriate `docker-compose.yml` for your edition:

**OpenSource Edition:**
```bash
curl -o docker-compose.yml https://your-registry.com/atlasclaw/docker-compose-opensource.yml
```

**Enterprise Edition:**
```bash
curl -o docker-compose.yml https://your-registry.com/atlasclaw/docker-compose-enterprise.yml
```

### 3. Create Configuration

Create `/opt/atlasclaw/config/atlasclaw.json`:

**OpenSource:**
```json
{
  "workspace": {
    "path": "./data"
  },
  "database": {
    "type": "sqlite",
    "sqlite": {
      "path": "./data/atlasclaw.db"
    }
  },
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
        "api_key": "YOUR_API_KEY_HERE",
        "api_type": "openai"
      }
    ]
  },
  "auth": {
    "provider": "api_key",
    "api_key": {
      "keys": {
        "sk-your-secret-key": {
          "user_id": "admin",
          "roles": ["admin"]
        }
      }
    }
  }
}
```

**Enterprise:**
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
      "password": "your-mysql-password",
      "charset": "utf8mb4"
    }
  },
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
        "api_key": "YOUR_API_KEY_HERE",
        "api_type": "openai"
      }
    ]
  },
  "auth": {
    "provider": "oidc",
    "oidc": {
      "issuer": "https://auth.your-company.com",
      "client_id": "atlasclaw-client",
      "client_secret": "your-client-secret",
      "redirect_uri": "https://atlasclaw.your-company.com/api/auth/callback"
    }
  }
}
```

Set proper permissions:
```bash
chmod 600 /opt/atlasclaw/config/atlasclaw.json
```

### 4. Start AtlasClaw

```bash
cd /opt/atlasclaw
docker compose up -d
```

### 5. Run Database Migrations (Enterprise Only)

```bash
docker compose exec atlasclaw alembic upgrade head
```

### 6. Verify Deployment

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{"status": "healthy", "timestamp": "2026-03-23T10:00:00+00:00"}
```

Access the web UI at: `http://your-server-ip:8000`

---

## Available Images

### OpenSource Edition

- **Image**: `your-registry.com/atlasclaw:latest`
- **Features**: SQLite database, single container
- **Best for**: Small teams, evaluation, development

### Enterprise Edition

- **Image**: `your-registry.com/atlasclaw-official:latest`
- **Features**: MySQL support, multi-container, high availability
- **Best for**: Production, large organizations

### Pull Images Manually

```bash
# OpenSource
docker pull your-registry.com/atlasclaw:latest

# Enterprise
docker pull your-registry.com/atlasclaw-official:latest
```

---

## Operations

### View Logs

```bash
docker compose logs -f atlasclaw
```

### Stop Services

```bash
docker compose down
```

### Update to Latest Version

```bash
# Pull latest images
docker compose pull

# Restart services
docker compose up -d

# Enterprise only: run migrations
docker compose exec atlasclaw alembic upgrade head
```

### Backup

**OpenSource:**
```bash
# Backup data directory
tar -czf atlasclaw-backup-$(date +%Y%m%d).tar.gz /opt/atlasclaw/data /opt/atlasclaw/config
```

**Enterprise:**
```bash
# Backup database
docker exec atlasclaw-mysql mysqldump -u root -p atlasclaw > atlasclaw-db-$(date +%Y%m%d).sql

# Backup files
tar -czf atlasclaw-backup-$(date +%Y%m%d).tar.gz /opt/atlasclaw/data /opt/atlasclaw/config
```

---

## Configuration Reference

### LLM Provider

Configure in `atlasclaw.json`:

```json
{
  "model": {
    "primary": "deepseek-main",
    "tokens": [
      {
        "id": "deepseek-main",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "your-api-key"
      }
    ]
  }
}
```

### Authentication

**API Key (OpenSource):**
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

**OIDC/OAuth2 (Enterprise):**
```json
{
  "auth": {
    "provider": "oidc",
    "oidc": {
      "issuer": "https://auth.company.com",
      "client_id": "your-client-id",
      "client_secret": "your-client-secret"
    }
  }
}
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs atlasclaw

# Verify config syntax
docker run --rm -v /opt/atlasclaw/config/atlasclaw.json:/app/atlasclaw.json:ro your-registry.com/atlasclaw:latest python -c "import json; json.load(open('/app/atlasclaw.json'))"
```

### Database Connection Failed (Enterprise)

```bash
# Check MySQL container
docker compose ps mysql
docker compose logs mysql

# Test MySQL connection
docker compose exec mysql mysql -u atlasclaw -p -e "SELECT 1"
```

### Port Already in Use

Edit `docker-compose.yml` to change the port mapping:

```yaml
ports:
  - "8080:8000"  # Change 8080 to your preferred port
```

### Permission Denied

Ensure proper file permissions:

```bash
chmod 600 /opt/atlasclaw/config/atlasclaw.json
chown -R $(id -u):$(id -g) /opt/atlasclaw/data
```

---

## Support

For technical support, contact your AtlasClaw representative or refer to the full documentation at [docs link].
