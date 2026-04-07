# AtlasClaw + SmartCMP SaaS Quick Start Guide

This guide walks you through deploying AtlasClaw from scratch and integrating it with the SmartCMP SaaS platform, enabling AI-powered conversations for viewing CMP pending approvals, approving/rejecting tickets, requesting resources, and more.

---

## 1. Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Linux / macOS / Windows |
| Python | 3.11+ |
| Git | 2.x |
| LLM API Key | A model that supports **Function Calling** (e.g., DeepSeek-Chat, GPT-4) |

> **Note**: Reasoning models like `deepseek-reasoner` do **not** support tool calling. Use `deepseek-chat` or similar models instead.

---

## 2. Clone Repositories

```bash
# Clone the main project
git clone https://github.com/CloudChef/atlasclaw.git
cd atlasclaw

# Clone the providers repository (must be in the same parent directory)
cd ..
git clone https://github.com/CloudChef/atlasclaw-providers.git
cd atlasclaw
```

Expected directory structure:

```
your-workspace/
├── atlasclaw/                  # Main project
└── atlasclaw-providers/        # Extensions (Providers, Skills, Channels)
    ├── providers/
    │   └── SmartCMP-Provider/  # SmartCMP integration
    ├── skills/
    └── channels/
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `wecom-aibot-sdk` fails to install, you can safely ignore it -- it does not affect SmartCMP functionality.

---

## 4. Configure Environment Variables

Copy `.env.example` to `.env` and update the key settings:

```bash
cp .env.example .env
```

Edit the `.env` file:

```ini
# ============================================
# LLM Model Configuration (Required)
# ============================================
# Example using DeepSeek-Chat (must support Function Calling)
TOKEN_2_PROVIDER=deepseek
TOKEN_2_MODEL=deepseek-chat
TOKEN_2_BASE_URL=https://api.deepseek.com
TOKEN_2_API_KEY=<your-deepseek-api-key>

LLM_TEMPERATURE=0.2

# ============================================
# SmartCMP SaaS Configuration (Required)
# ============================================
CMP_URL=https://console.smartcmp.cloud
CMP_USERNAME=your-cmp-username
CMP_PASSWORD=your-cmp-password

# ============================================
# Authentication Configuration
# ============================================
ATLASCLAW_JWT_SECRET=change-me-to-a-secure-secret-key
```

### Configuration Reference

| Variable | Description | Example |
|----------|-------------|----------|
| `TOKEN_2_MODEL` | LLM model name, **must support Function Calling** | `deepseek-chat` |
| `TOKEN_2_API_KEY` | LLM API Key | `sk-xxx...` |
| `CMP_URL` | SmartCMP platform URL | `https://console.smartcmp.cloud` |
| `CMP_USERNAME` | SmartCMP login username | `user@company.com` |
| `CMP_PASSWORD` | SmartCMP login password | `your-password` |

---

## 5. Start the Service

```bash
uvicorn app.atlasclaw.main:app --host 0.0.0.0 --port 8000
```

On successful startup, the logs will show:

```
[AtlasClaw] Agent created with model: openai/deepseek-chat
[AtlasClaw] Skills loaded: 30 executable, 19 markdown
[AtlasClaw] Application started successfully
```

> Verify that **Skills loaded** shows an executable count > 13 (which includes the SmartCMP script tools).

To run in the background (Linux):

```bash
nohup uvicorn app.atlasclaw.main:app --host 0.0.0.0 --port 8000 > atlasclaw.log 2>&1 &
```

---

## 6. Access the Web UI

Open your browser and navigate to:

```
http://<server-ip>:8000
```

Default login credentials:

| Username | Password |
|----------|----------|
| `admin` | `admin` |

> It is recommended to change the password after the first login.

---

## 7. Test: View CMP Pending Approvals

After logging in, enter the following message in the chat interface:

```
List CMP pending approvals
```

The agent will automatically:
1. Recognize the intent and invoke the `smartcmp_list_pending` tool
2. Authenticate with the SmartCMP platform using the configured CMP credentials
3. Retrieve the pending tickets and display them in a table

### Expected Response Example

The agent will return something like:

```
Pending Approvals - 3 items

| # | Priority | Name                                  | Ticket ID         | Type            | Requester      | Waiting   |
|---|----------|---------------------------------------|-------------------|-----------------|----------------|-----------|
| 1 | High     | Test ticket for build verification    | TIC20260316000001 | Incident Ticket | user@email.com | 431 hours |
| 2 | High     | Urgent request                        | TIC20260313000006 | Problem Ticket  | user@email.com | 503 hours |
| 3 | High     | N/A                                   | TIC20260313000004 | Problem Ticket  | user@email.com | 504 hours |
```

### More Conversation Examples

```
# Approve a ticket
Approve ticket TIC20260316000001

# Reject a ticket
Reject ticket TIC20260313000006, reason: incomplete information

# View service catalog
List available CMP services

# Query cloud resource costs
Show cloud resource cost optimization suggestions for this month
```

---

## 8. Troubleshooting

### Q: The agent searches the web instead of calling SmartCMP tools?

**Cause**: The configured model does not support Function Calling (e.g., `deepseek-reasoner`).

**Solution**: Ensure the model in `.env` supports tool calling:
```ini
TOKEN_2_MODEL=deepseek-chat    # Supports Function Calling
# TOKEN_1_MODEL=deepseek-reasoner  # Not supported
```

Also verify that `model.primary` in `atlasclaw.json` points to the correct token:
```json
"model": {
  "primary": "model-2"
}
```

### Q: SmartCMP returns a 401 authentication error?

**Check**:
1. Verify `CMP_USERNAME` and `CMP_PASSWORD` in `.env` are correct
2. Confirm the account can log in at https://console.smartcmp.cloud
3. Ensure you are using the latest code (which includes the password environment variable fix)

### Q: Installation fails with `wecom-aibot-sdk` error?

This can be safely ignored, or install with:
```bash
pip install -r requirements.txt --ignore-installed wecom-aibot-sdk || true
```

---

## 9. Docker Deployment (Optional)

If you prefer not to install a Python environment, you can deploy with Docker:

```bash
# Prepare directories
mkdir -p /opt/atlasclaw/{workspace,data,extensions/{providers,skills,channels}}

# Copy configuration files
cp atlasclaw.json /opt/atlasclaw/workspace/
cp .env /opt/atlasclaw/workspace/

# Copy extensions
cp -r ../atlasclaw-providers/providers/* /opt/atlasclaw/extensions/providers/
cp -r ../atlasclaw-providers/skills/* /opt/atlasclaw/extensions/skills/

# Start
cd build
docker-compose up -d
```

Access `http://<server-ip>:8000` to start using the application.

---

## 10. Support

If you encounter any issues, please provide the following information:
- The `Skills loaded` and `Agent created with model` lines from the startup logs
- Model configuration from `.env` (with API keys redacted)
- Error messages from the browser developer console (F12)
