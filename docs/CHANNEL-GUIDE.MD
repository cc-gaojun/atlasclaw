# AtlasClaw Channel Guide

## Overview

AtlasClaw Channel system provides a unified way to integrate with various messaging platforms. This guide explains the architecture, how to use channels, and provides a detailed Feishu integration example.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AtlasClaw Channel Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │   ChannelHandler │     │  ChannelManager │     │    ChannelStore     │   │
│  │    (Interface)   │────▶│   (Lifecycle)   │────▶│  (User Config)      │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────────┘   │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │  Built-in       │     │  Connection     │     │  users/{user_id}/   │   │
│  │  Handlers       │     │  State Machine  │     │  channels/{type}.json│   │
│  │                 │     │                 │     │                     │   │
│  │  • WebSocket    │     │  • CONNECTING   │     │  • Connection list  │   │
│  │  • SSE          │     │  • CONNECTED    │     │  • Config per conn  │   │
│  │  • REST         │     │  • DISCONNECTED │     │  • Enable/disable   │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Extension Channels (Providers)                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │   Feishu    │  │    Slack    │  │  WhatsApp   │  │    ...     │ │   │
│  │  │  (WebSocket)│  │ (SocketMode)│  │ (WebSocket) │  │            │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ChannelHandler Interface

All channels implement the `ChannelHandler` interface:

**Lifecycle Methods:**
- `setup()` - Initialize with configuration
- `start()` - Start the channel
- `connect()` - Establish long connection (for WebSocket/Socket mode)
- `disconnect()` - Close long connection
- `stop()` - Stop and cleanup

**Message Methods:**
- `handle_inbound()` - Process incoming messages
- `send_message()` - Send outgoing messages

**Configuration Methods:**
- `validate_config()` - Validate connection settings
- `describe_schema()` - Return configuration schema

### Connection Modes

| Mode | Direction | Use Case | Examples |
|------|-----------|----------|----------|
| **Long Connection** | AtlasClaw → Platform | Real-time messaging | Feishu WebSocket, Slack Socket Mode |
| **Webhook** | Platform → AtlasClaw | HTTP callbacks | Generic webhooks |
| **Server** | Client → AtlasClaw | Built-in channels | WebSocket, SSE, REST |

### Connection States

```
DISCONNECTED → CONNECTING → CONNECTED
                    ↑           │
                    └───────────┘ (auto-reconnect on failure)
```

## How to Use

### Step 1: Choose Channel Type

**Built-in Channels** (no provider needed):
- `websocket` - For browser/app real-time communication
- `sse` - For server-sent events
- `rest` - For HTTP-based integration

**Extension Channels** (require provider):
- `feishu` - Feishu/Lark messaging
- `slack` - Slack messaging
- Custom providers in `providers/{name}/channels/`

### Step 2: Create Connection Configuration

Create a configuration file at:
```
users/{user_id}/channels/{channel_type}.json
```

Or store user-level channel config in `users/{user_id}/user_setting.json`:
```json
{
  "channels": {
    "feishu": {
      "connections": [
        {
          "id": "unique-connection-id",
          "name": "Display Name",
          "enabled": true,
          "config": {
            "app_id": "cli_xxx",
            "app_secret": "xxx"
          }
        }
      ]
    }
  }
}
```

Configuration structure (standalone file):
```json
{
  "channel_type": "{type}",
  "connections": [
    {
      "id": "unique-connection-id",
      "name": "Display Name",
      "enabled": true,
      "is_default": false,
      "config": {
        // Channel-specific configuration
      }
    }
  ]
}
```

### Step 3: Enable Connection

Use the REST API to enable the connection:

```bash
POST /api/channels/{type}/connections/{id}/enable
```

Or use ChannelManager programmatically:

```python
from app.atlasclaw.channels import ChannelManager

manager = ChannelManager(workspace_path)
await manager.enable_connection(user_id, channel_type, connection_id)
```

### Step 4: Handle Messages

For **long connection** channels:
- Messages are received via callback
- Automatically routed to SessionManager

For **webhook** channels:
- Platform sends POST to `/api/channel-hooks/{type}/{connection_id}`
- Handler processes and routes to SessionManager

### Step 5: Send Messages

```python
from app.atlasclaw.channels import OutboundMessage

outbound = OutboundMessage(
    chat_id="chat-id",
    content="Hello!",
    content_type="text"
)

result = await handler.send_message(outbound)
```

## API Reference

### Channel Management APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/channels` | List all channel types |
| GET | `/api/channels/{type}/schema` | Get configuration schema |
| GET | `/api/channels/{type}/connections` | List user's connections |
| POST | `/api/channels/{type}/connections` | Create new connection |
| PATCH | `/api/channels/{type}/connections/{id}` | Update connection |
| DELETE | `/api/channels/{type}/connections/{id}` | Delete connection |
| POST | `/api/channels/{type}/connections/{id}/verify` | Verify configuration |
| POST | `/api/channels/{type}/connections/{id}/enable` | Enable connection |
| POST | `/api/channels/{type}/connections/{id}/disable` | Disable connection |

### Webhook Endpoint

For platforms using webhook mode:

```
POST /api/channel-hooks/{channel_type}/{connection_id}
```

## Feishu Integration Example

### Overview

Feishu (Lark) integration uses **long connection mode** via WebSocket. AtlasClaw establishes a persistent connection to Feishu's Event Center to receive real-time messages.

### Architecture

```
┌─────────────┐      WebSocket       ┌─────────────┐      HTTP API      ┌─────────────┐
│   User      │◀────────────────────▶│   Feishu    │◀─────────────────▶│  AtlasClaw  │
│   (App)     │                      │   Server    │                     │   Server    │
└─────────────┘                      └─────────────┘                     └─────────────┘
                                                                              │
                                                                              ▼
                                                                       ┌─────────────┐
                                                                       │  FeishuHandler│
                                                                       │  (WebSocket)  │
                                                                       └─────────────┘
```

### Step 1: Create Feishu Application

1. Visit [Feishu Open Platform](https://open.feishu.cn/)
2. Click "Create Application" → "Custom App"
3. Fill in app name and description
4. Note the **App ID** and **App Secret**

### Step 2: Enable Capabilities

1. Go to "Add App Capability"
2. Select "Bot" to enable bot functionality
3. Configure bot name and avatar

### Step 3: Configure Permissions

Required permissions:
- `im:chat:readonly` - Read group info
- `im:message:send_as_bot` - Send messages
- `im:message.group_msg` - Receive group messages
- `im:message.p2p_msg` - Receive private messages
- `event:message` - Event subscription

### Step 4: Publish Application

1. Go to "Version Management"
2. Create version and submit
3. Install in your workspace

### Step 5: Create AtlasClaw Connection

Create file: `users/{user_id}/channels/feishu.json`

```json
{
  "channel_type": "feishu",
  "connections": [
    {
      "id": "feishu-bot-001",
      "name": "My Feishu Bot",
      "enabled": true,
      "is_default": true,
      "config": {
        "app_id": "cli_xxxxxxxxxxxxxxxx",
        "app_secret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "encrypt_key": "",
        "verification_token": ""
      }
    }
  ]
}
```

### Step 6: Enable Connection

```bash
curl -X POST http://localhost:8000/api/channels/feishu/connections/feishu-bot-001/enable
```

Or programmatically:

```python
from app.atlasclaw.channels import ChannelManager

manager = ChannelManager("/path/to/workspace")
await manager.enable_connection("user-123", "feishu", "feishu-bot-001")
```

### Connection Flow

When enabled, AtlasClaw performs these steps:

1. **Authenticate** - Uses App ID and App Secret to get access token
2. **Connect** - Establishes WebSocket connection to Feishu Event Center
3. **Subscribe** - Subscribes to message events
4. **Listen** - Continuously receives events via WebSocket
5. **Process** - Converts events to InboundMessage and routes to Agent
6. **Reply** - Sends responses via Feishu HTTP API

### Message Flow

**Incoming Message:**
```
User sends message in Feishu
    ↓
Feishu Server → WebSocket → FeishuHandler
    ↓
Convert to InboundMessage
    ↓
Route to SessionManager → Agent
    ↓
Generate response
```

**Outgoing Message:**
```
Agent generates response
    ↓
FeishuHandler.send_message()
    ↓
Convert to Feishu format
    ↓
POST to Feishu API
    ↓
User receives reply
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| app_id | Yes | Feishu application ID (cli_xxx) |
| app_secret | Yes | Feishu application secret |
| encrypt_key | No | Message encryption key |
| verification_token | No | Webhook verification token |

### Troubleshooting

**Connection fails:**
- Verify App ID and App Secret are correct
- Check app has required permissions
- Ensure app is published and installed

**Messages not received:**
- Check WebSocket connection status
- Verify bot is added to chat/group
- Review AtlasClaw logs for errors

**Cannot send replies:**
- Verify `im:message:send_as_bot` permission
- Check access token is valid
- Ensure app is not rate-limited

### Advanced: Custom Feishu Handler

To customize Feishu integration, create a provider:

```
providers/
└── my_feishu/
    ├── channels/
    │   └── feishu_custom.py
    └── config.json
```

The custom handler inherits from ChannelHandler and overrides:
- `connect()` - Custom WebSocket connection logic
- `handle_inbound()` - Custom message parsing
- `send_message()` - Custom reply formatting

---

## Summary

AtlasClaw Channel system provides:

1. **Unified Interface** - All channels implement ChannelHandler
2. **Long Connection Support** - WebSocket/Socket mode for real-time messaging
3. **User Isolation** - Each user has independent channel configurations
4. **Flexible Architecture** - Easy to add new platform integrations
5. **State Management** - Connection lifecycle with auto-reconnect

For Feishu and other platforms, the integration follows the same pattern: configure credentials, enable connection, and the system handles message routing automatically.
