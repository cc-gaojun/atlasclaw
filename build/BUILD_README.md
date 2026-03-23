# AtlasClaw Build

This directory contains build scripts and configurations for both OpenSource and Enterprise editions of AtlasClaw.

## Quick Comparison

| Feature | OpenSource | Enterprise |
|---------|------------|------------|
| Database | SQLite (built-in) | MySQL 8.5 |
| Deployment | Single container | Multi-container |
| Best for | Development / Small teams | Production / Large organizations |
| Resources | Minimal | Configurable limits |
| High Availability | No | Yes (with external MySQL) |

## Files

### Dockerfiles

| File | Description |
|------|-------------|
| `Dockerfile.opensource` | Lightweight build with SQLite |
| `Dockerfile.enterprise` | Multi-stage build with MySQL support |

### Compose Files

| File | Description |
|------|-------------|
| `docker-compose.opensource.yml` | Single AtlasClaw container |
| `docker-compose.enterprise.yml` | AtlasClaw + MySQL 8.5 + Secrets |

### Scripts

| File | Description |
|------|-------------|
| `build.sh` | Automated build script with mode selection |

## Usage

### Build Options

```bash
./build.sh --mode opensource|enterprise [--tag VERSION] [--repo REGISTRY]
```

| Option | Description |
|--------|-------------|
| `--mode opensource` | Build OpenSource edition (image: `atlasclaw`) |
| `--mode enterprise` | Build Enterprise edition (image: `atlasclaw-official`) |
| `--tag` | Version tag (default: `latest`) |
| `--repo` | Docker registry prefix (optional, e.g., `registry.example.com`) |

### OpenSource Edition

```bash
# Build locally
./build.sh --mode opensource --tag v1.0.0

# Build with registry prefix
./build.sh --mode opensource --tag v1.0.0 --repo registry.example.com
# Creates image: registry.example.com/atlasclaw:v1.0.0
```

**Features:**
- Image name: `atlasclaw`
- Single Docker container
- SQLite database (auto-created)
- Minimal resource usage
- Quick start

**Deploy:**
```bash
cd build
docker-compose up -d
```

### Enterprise Edition

```bash
# Build locally
./build.sh --mode enterprise --tag v1.0.0

# Build with registry prefix
./build.sh --mode enterprise --tag v1.0.0 --repo registry.example.com
# Creates image: registry.example.com/atlasclaw-official:v1.0.0
```

**Features:**
- Image name: `atlasclaw-official`
- MySQL 8.5 LTS database
- Docker secrets for passwords
- Resource limits (4 CPU / 8GB RAM)
- Health checks
- Persistent volumes

**Deploy:**
```bash
cd build
docker-compose up -d
docker-compose exec atlasclaw alembic upgrade head
```

## Build Script

The `build.sh` script automates:

1. **Prerequisites check** - Docker, Docker Compose
2. **Python validation** - Installs dependencies locally to verify
3. **Configuration generation** - Creates `atlasclaw.json`
4. **Secret generation** (Enterprise) - Auto-generates MySQL passwords
5. **Docker build** - Builds the appropriate image
6. **Cleanup** - Removes temporary files

### Generated Files

After running build script:

```
build/
├── config/
│   └── atlasclaw.json          # Main configuration
├── secrets/                    # Enterprise only
│   ├── mysql_root_password.txt
│   └── mysql_password.txt
├── data/                       # SQLite/Volume data
├── logs/                       # Application logs
├── mysql-data/                 # Enterprise MySQL data
└── docker-compose.yml -> docker-compose.{mode}.yml
```

## Configuration

### OpenSource

Edit `config/atlasclaw.json`:

```json
{
  "database": {
    "type": "sqlite",
    "sqlite": {
      "path": "./data/atlasclaw.db"
    }
  }
}
```

### Enterprise

Edit `config/atlasclaw.json`:

```json
{
  "database": {
    "type": "mysql",
    "mysql": {
      "host": "mysql",
      "port": 3306,
      "database": "atlasclaw",
      "user": "atlasclaw",
      "password": "auto-generated",
      "charset": "utf8mb4"
    }
  }
}
```

**Passwords are auto-generated in `secrets/` directory.**

## Operations

### View Logs

```bash
docker-compose logs -f atlasclaw
docker-compose logs -f mysql    # Enterprise only
```

### Stop

```bash
docker-compose down
```

### Backup

**OpenSource:**
```bash
tar -czf backup.tar.gz config/ data/ logs/
```

**Enterprise:**
```bash
# Backup database
docker exec atlasclaw-mysql mysqldump -u root -p atlasclaw > db_backup.sql

# Backup files
tar -czf backup.tar.gz config/ data/ logs/
```

## Troubleshooting

### Port Already in Use

Edit `docker-compose.yml`:

```yaml
ports:
  - "8080:8000"
```

### Permission Denied

```bash
chmod 600 config/atlasclaw.json
chmod 600 secrets/*.txt  # Enterprise
```

### Build Failures

```bash
# Clear Docker cache
docker builder prune

# Retry build
./build.sh --mode {opensource|enterprise}
```
