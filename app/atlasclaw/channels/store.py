# -*- coding: utf-8 -*-
"""Channel configuration store for user-specific channel connections."""

from __future__ import annotations

import json
import logging
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ChannelConnection

logger = logging.getLogger(__name__)

# Default user setting structure
DEFAULT_USER_SETTING = {
    "channels": {},
    "preferences": {}
}


class ChannelStore:
    """Store for user channel connection configurations.
    
    Stores channel configurations per-user in:
    <workspace>/users/<user_id>/user_setting.json
    
    The channels are stored under the "channels" key with structure:
    {
        "channels": {
            "<channel_type>": {
                "connections": [...]
            }
        },
        "preferences": {...}
    }
    
    Note: providers are system-level configuration, not user-level.
    """
    
    def __init__(self, workspace_path: Path):
        """Initialize channel store.
        
        Args:
            workspace_path: Path to workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self.users_dir = self.workspace_path / "users"
    
    def _get_user_setting_path(self, user_id: str) -> Path:
        """Get user's setting file path.
        
        Args:
            user_id: User identifier
            
        Returns:
            Path to user_setting.json
        """
        return self.users_dir / user_id / "user_setting.json"
    
    def _load_user_setting(self, user_id: str) -> Dict[str, Any]:
        """Load user setting from file.
        
        Args:
            user_id: User identifier
            
        Returns:
            User setting dictionary
        """
        setting_path = self._get_user_setting_path(user_id)
        
        if not setting_path.exists():
            # Use deepcopy to avoid modifying the original DEFAULT_USER_SETTING
            return copy.deepcopy(DEFAULT_USER_SETTING)
        
        try:
            with open(setting_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Ensure all required keys exist
            if "channels" not in data:
                data["channels"] = {}
            if "preferences" not in data:
                data["preferences"] = {}
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load user setting for {user_id}: {e}")
            return copy.deepcopy(DEFAULT_USER_SETTING)
    
    def _save_user_setting(self, user_id: str, setting: Dict[str, Any]) -> bool:
        """Save user setting to file.
        
        Args:
            user_id: User identifier
            setting: User setting dictionary
            
        Returns:
            True if saved successfully
        """
        try:
            setting_path = self._get_user_setting_path(user_id)
            setting_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(setting_path, "w", encoding="utf-8") as f:
                json.dump(setting, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user setting for {user_id}: {e}")
            return False
    
    def get_connections(
        self,
        user_id: str,
        channel_type: str
    ) -> List[ChannelConnection]:
        """Get all connections for a user and channel type.
        
        Args:
            user_id: User identifier
            channel_type: Channel type
            
        Returns:
            List of channel connections
        """
        setting = self._load_user_setting(user_id)
        channel_data = setting.get("channels", {}).get(channel_type, {})
        connections_data = channel_data.get("connections", [])
        
        connections = []
        for conn_data in connections_data:
            connection = ChannelConnection(
                id=conn_data.get("id", ""),
                name=conn_data.get("name", ""),
                channel_type=channel_type,
                config=conn_data.get("config", {}),
                enabled=conn_data.get("enabled", True),
                is_default=conn_data.get("is_default", False),
            )
            connections.append(connection)
        
        return connections
    
    def get_connection(
        self,
        user_id: str,
        channel_type: str,
        connection_id: str
    ) -> Optional[ChannelConnection]:
        """Get a specific connection.
        
        Args:
            user_id: User identifier
            channel_type: Channel type
            connection_id: Connection identifier
            
        Returns:
            Channel connection or None if not found
        """
        connections = self.get_connections(user_id, channel_type)
        
        for conn in connections:
            if conn.id == connection_id:
                return conn
        
        return None
    
    def save_connection(
        self,
        user_id: str,
        channel_type: str,
        connection: ChannelConnection
    ) -> bool:
        """Save a connection to user_setting.json.
        
        Args:
            user_id: User identifier
            channel_type: Channel type
            connection: Connection to save
            
        Returns:
            True if saved successfully
        """
        try:
            setting = self._load_user_setting(user_id)
            
            # Ensure channels structure exists
            if "channels" not in setting:
                setting["channels"] = {}
            if channel_type not in setting["channels"]:
                setting["channels"][channel_type] = {"connections": []}
            
            # Get current connections
            connections = setting["channels"][channel_type].get("connections", [])
            
            # Update or add connection
            found = False
            for i, conn_data in enumerate(connections):
                if conn_data.get("id") == connection.id:
                    connections[i] = {
                        "id": connection.id,
                        "name": connection.name,
                        "config": connection.config,
                        "enabled": connection.enabled,
                        "is_default": connection.is_default,
                    }
                    found = True
                    break
            
            if not found:
                connections.append({
                    "id": connection.id,
                    "name": connection.name,
                    "config": connection.config,
                    "enabled": connection.enabled,
                    "is_default": connection.is_default,
                })
            
            # Update setting
            setting["channels"][channel_type]["connections"] = connections
            
            return self._save_user_setting(user_id, setting)
            
        except Exception as e:
            logger.error(f"Failed to save connection: {e}")
            return False
    
    def delete_connection(
        self,
        user_id: str,
        channel_type: str,
        connection_id: str
    ) -> bool:
        """Delete a connection from user_setting.json.
        
        Args:
            user_id: User identifier
            channel_type: Channel type
            connection_id: Connection identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            setting = self._load_user_setting(user_id)
            
            if channel_type not in setting.get("channels", {}):
                return False
            
            connections = setting["channels"][channel_type].get("connections", [])
            
            # Remove connection
            connections = [c for c in connections if c.get("id") != connection_id]
            
            # Update setting
            setting["channels"][channel_type]["connections"] = connections
            
            # Remove channel type if no connections left
            if not connections:
                del setting["channels"][channel_type]
            
            return self._save_user_setting(user_id, setting)
            
        except Exception as e:
            logger.error(f"Failed to delete connection: {e}")
            return False
    
    def update_connection_status(
        self,
        user_id: str,
        channel_type: str,
        connection_id: str,
        enabled: bool
    ) -> bool:
        """Update connection enabled status.
        
        Args:
            user_id: User identifier
            channel_type: Channel type
            connection_id: Connection identifier
            enabled: New enabled status
            
        Returns:
            True if updated successfully
        """
        connection = self.get_connection(user_id, channel_type, connection_id)
        
        if not connection:
            return False
        
        connection.enabled = enabled
        return self.save_connection(user_id, channel_type, connection)
