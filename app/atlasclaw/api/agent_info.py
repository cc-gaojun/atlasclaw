# -*- coding: utf-8 -*-
"""Agent information API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/agent", tags=["agent"])


class AgentInfoResponse(BaseModel):
    """Agent information response."""
    name: str
    description: str
    welcome_message: str
    soul: Dict[str, Any]


@router.get("/info", response_model=AgentInfoResponse)
async def get_agent_info() -> AgentInfoResponse:
    """Get main agent information including welcome message from SOUL.md.
    
    Returns:
        Agent information with welcome message
    """
    try:
        # Load main agent SOUL.md
        soul_path = Path(".atlasclaw/agents/main/SOUL.md")
        identity_path = Path(".atlasclaw/agents/main/IDENTITY.md")
        
        soul_data = _parse_soul_md(soul_path.read_text(encoding="utf-8")) if soul_path.exists() else {}
        identity_data = _parse_identity_md(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else {}
        
        # Build welcome message from SOUL.md
        welcome_parts = []
        
        # Add name
        name = soul_data.get("name", identity_data.get("name", "AtlasClaw"))
        welcome_parts.append(f"Hello! I'm {name}.")
        
        # Add description
        description = soul_data.get("description", identity_data.get("description", ""))
        if description:
            welcome_parts.append(description)
        
        # Add core values if available
        core_values = soul_data.get("core_values", [])
        if core_values:
            welcome_parts.append(f"\nMy core values: {', '.join(core_values)}")
        
        welcome_message = "\n\n".join(welcome_parts)
        
        return AgentInfoResponse(
            name=name,
            description=description,
            welcome_message=welcome_message,
            soul=soul_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load agent info: {str(e)}")


def _parse_soul_md(content: str) -> Dict[str, Any]:
    """Parse SOUL.md content.
    
    Args:
        content: SOUL.md file content
        
    Returns:
        Parsed data dictionary
    """
    data = {}
    current_section = None
    current_list = []
    
    for line in content.split("\n"):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Parse headers
        if line.startswith("# "):
            data["name"] = line[2:].strip()
        elif line.startswith("## "):
            # Save previous section if it was a list
            if current_section and current_list:
                data[current_section] = current_list
                current_list = []
            
            current_section = line[3:].strip().lower().replace(" ", "_")
        
        # Parse list items
        elif line.startswith("- "):
            item = line[2:].strip()
            if current_section:
                current_list.append(item)
        
        # Parse key-value pairs
        elif ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            data[key] = value
    
    # Save last section if it was a list
    if current_section and current_list:
        data[current_section] = current_list
    
    return data


def _parse_identity_md(content: str) -> Dict[str, Any]:
    """Parse IDENTITY.md content.
    
    Args:
        content: IDENTITY.md file content
        
    Returns:
        Parsed data dictionary
    """
    data = {}
    
    for line in content.split("\n"):
        line = line.strip()
        
        if line.startswith("# "):
            data["name"] = line[2:].strip()
        elif line.startswith("## "):
            pass  # Section headers
        elif ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            data[key] = value
    
    return data
