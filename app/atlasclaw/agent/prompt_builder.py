"""
System prompt builder.

This module builds the runtime system prompt for each agent loop. It supports
three modes:

- `full`: include the complete runtime prompt.
- `minimal`: omit optional sections used mainly by the primary agent.
- `none`: emit only the minimal identity line.

In full mode, the prompt is assembled in a stable order so the runtime can
reason about tools, safety constraints, skills, workspace context, optional
reply conventions, and runtime metadata consistently.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.atlasclaw.auth.models import UserInfo


class PromptMode(Enum):
    """System prompt mode"""
    FULL = "full"          # include all sections
    MINIMAL = "minimal"    # Sub-agent mode without optional runtime sections
    NONE = "none"          # only the basic identity line


@dataclass
class SandboxConfig:
    """Sandbox configuration"""
    enabled: bool = False
    mode: str = "off"          # off | agent | session
    workspace_root: str = ""
    elevated_exec: bool = False


@dataclass
class PromptBuilderConfig:
    """PromptBuilder configuration"""
    mode: PromptMode = PromptMode.FULL
    bootstrap_max_chars: int = 20000
    workspace_path: str = "~/.atlasclaw/workspace"
    user_timezone: Optional[str] = None
    time_format: str = "auto"  # auto | 12 | 24
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    agent_name: str = "AtlasClaw"
    agent_description: str = "Enterprise AI Assistant"
    # Marker file used to detect a newly initialized workspace
    new_workspace_marker: str = ".atlasclaw_new_workspace"
    # Total character budget for the Markdown skill index section
    md_skills_max_index_chars: int = 3000
    # Maximum description length for a single Markdown skill entry
    md_skills_desc_max_chars: int = 200
    # Maximum number of Markdown skill entries in the index section
    md_skills_max_count: int = 20


class PromptBuilder:
    """
    System prompt builder.

    Fully overrides every section in `system-prompt.md`.

    Example usage:
        ```python
        config = PromptBuilderConfig(
            mode=PromptMode.FULL,
            workspace_path="~/.atlasclaw/workspace",
        )
        builder = PromptBuilder(config)
        
        system_prompt = builder.build(
            session=session,
            skills=[{"name":"query_vm", "description":"query virtual machines"}],
            tools=[{"name":"read_file", "description":"read file"}],
        )
        ```
    """

    # Bootstrap files injected into the prompt when available
    BOOTSTRAP_FILES = [
        "AGENTS.md",
        "SOUL.md",
        "TOOLS.md",
        "IDENTITY.md",
        "USER.md",
        "HEARTBEAT.md",
        "BOOTSTRAP.md",
    ]
    
    def __init__(self, config: PromptBuilderConfig):
        """Initialize the prompt builder with the provided configuration."""
        self.config = config
    
    def build(
        self,
        session: Optional[object] = None,
        skills: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        md_skills: Optional[list[dict]] = None,
        target_md_skill: Optional[dict] = None,
        user_info: Optional["UserInfo"] = None,
        provider_contexts: Optional[dict[str, dict]] = None,
    ) -> str:
        """
        Build the full system prompt for the current run.

        Args:
            session: Optional session metadata for the active conversation.
            skills: Executable skill metadata available to the agent.
            tools: Tool metadata exposed to the current agent.
            md_skills: Optional Markdown skill snapshot for prompt injection.
            provider_contexts: Optional provider LLM context for skill discovery.

        Returns:
            The assembled system prompt text.
        """
        if self.config.mode == PromptMode.NONE:
            return f"You are {self.config.agent_name}, {self.config.agent_description}."
        
        parts = []
        
        # 1. Core identity
        parts.append(self._build_identity())
        
        # 2. Tooling section
        if tools:
            parts.append(self._build_tooling(tools))
        
        # 3. Safety section
        parts.append(self._build_safety())
        
        # 3b. User context (when authenticated)
        if user_info and user_info.user_id not in ("anonymous", ""):
            user_ctx = self._build_user_context(user_info)
            if user_ctx:
                parts.append(user_ctx)
        
        if self.config.mode == PromptMode.FULL:
            # 4. Markdown skill index (HIGHEST PRIORITY - check these first!)
            if md_skills:
                md_index = self._build_md_skills_index(md_skills, provider_contexts)
                if md_index:
                    parts.append(md_index)
            
            # 4b. Executable skills (fallback only if no MD skill matches)
            if skills:
                parts.append(self._build_skills_listing(skills))

            if target_md_skill:
                parts.append(self._build_target_md_skill(target_md_skill))
            
            # 5. Self-update instructions
            parts.append(self._build_self_update())
            
            # 6. Workspace context
            parts.append(self._build_workspace_info())
            
            # 7. Documentation pointers
            parts.append(self._build_documentation())
            
            # 8. Project bootstrap context
            bootstrap = self._build_bootstrap()
            if bootstrap:
                parts.append(bootstrap)
            
            # 9. Reply-tag syntax
            reply_tags = self._build_reply_tags()
            if reply_tags:
                parts.append(reply_tags)
            
            # 12. Heartbeats(heartbeat)
            heartbeats = self._build_heartbeats()
            if heartbeats:
                parts.append(heartbeats)
        
        # 9. Sandbox(, when enabled)
        if self.config.sandbox.enabled:
            parts.append(self._build_sandbox())
        
        # 10. Current Date & Time()
        parts.append(self._build_datetime())
        
        # 13. Runtime(runtime information)
        parts.append(self._build_runtime_info())
        
        return "\n\n".join(p for p in parts if p)

    def _build_target_md_skill(self, target_md_skill: dict[str, str]) -> str:
        """Build a focused section for webhook-directed markdown skill execution."""
        qualified_name = target_md_skill.get("qualified_name", "")
        file_path = target_md_skill.get("file_path", "")
        provider = target_md_skill.get("provider", "")
        lines = ["## Target Markdown Skill", ""]
        if qualified_name:
            lines.append(f"Qualified name: {qualified_name}")
        if provider:
            lines.append(f"Provider: {provider}")
        if file_path:
            lines.append(f"File path: {file_path}")
        lines.append("You must execute only this markdown skill for the current run.")
        lines.append("Read the SKILL.md first before taking any other action.")
        return "\n".join(lines)
    
    def _build_user_context(self, user_info: "UserInfo") -> str:
        """Build a user identity section for the current authenticated operator."""
        lines = ["## Current User", ""]
        if user_info.display_name:
            lines.append(f"Name: {user_info.display_name}")
        lines.append(f"User ID: {user_info.user_id}")
        if user_info.tenant_id and user_info.tenant_id != "default":
            lines.append(f"Tenant: {user_info.tenant_id}")
        if user_info.roles:
            lines.append(f"Roles: {', '.join(user_info.roles)}")
        return "\n".join(lines)

    def _build_identity(self) -> str:
        """Build the identity section."""
        return f"""## Identity

You are {self.config.agent_name}, {self.config.agent_description}.

Your core capabilities include:
- Handling complex multi-turn conversations with context continuity
- Invoking various business skills (cloud resource management, ITSM, ticket processing, etc.)
- Managing long-term memory with semantic retrieval
- Supporting multi-step workflows and task collaboration"""
    
    def _build_tooling(self, tools: list[dict]) -> str:
        """Build the tool listing section."""
        lines = ["## Tools", ""]
        lines.append("You can use the following tools to complete tasks:")
        lines.append("")
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            lines.append(f"- **{name}**: {description}")
        return "\n".join(lines)
    
    def _build_safety(self) -> str:
        """Build the safety section."""
        return """## Safety

Please follow these safety guidelines:
- Avoid power-seeking behaviors or bypassing oversight
- Do not execute operations that may cause irreversible damage
- Sensitive information must be desensitized
- Respect user data privacy"""
    
    def _build_skills_listing(self, skills: list[dict]) -> str:
        """buildavailable Skills list(XML format)"""
        if not skills:
            return ""
        
        lines = ["## Built-in Tools (Use ONLY if no MD Skill matches)", "", "<available_skills>"]
        for s in skills:
            name = s.get("name", "unknown")
            description = s.get("description", "")
            location = s.get("location", "built-in")
            category = s.get("category", "utility")
            lines.append(f"""  <skill>
    <name>{name}</name>
    <description>{description}</description>
    <category>{category}</category>
    <location>{location}</location>
  </skill>""")
        lines.append("</available_skills>")
        lines.append("\nNOTE: These built-in tools are fallback options. ALWAYS check MD Skills section above first.")
        return "\n".join(lines)
    
    def _build_md_skills_index(
        self,
        md_skills: list[dict],
        provider_contexts: Optional[dict[str, dict]] = None,
    ) -> str:
        """
        Build the Markdown skills XML index section with provider context.

        Groups skills by provider and includes LLM context fields for better
        skill selection. Provider context (keywords, capabilities, use_when,
        avoid_when) helps LLM understand when to use each provider.

        Args:
            md_skills: MD Skills snapshot list
            provider_contexts: Optional provider LLM context dictionary

        Returns:
            index section (returns "" if empty)
        """
        if not md_skills:
            return ""

        max_count = self.config.md_skills_max_count
        desc_max = self.config.md_skills_desc_max_chars
        budget = self.config.md_skills_max_index_chars
        home_prefix = str(Path.home())
        provider_contexts = provider_contexts or {}

        # Execution instructions
        instructions = (
            "When a user's task matches a skill description below:\n"
            "1. Use the `read` tool to load the SKILL.md at the given location\n"
            "2. Follow the instructions in the SKILL.md to execute the task\n"
            "3. Use the available tools (exec, read, write, edit, etc.) as needed\n\n"
            "SKILL SELECTION GUIDANCE:\n"
            "- Check 'use_when' conditions to confirm the skill applies\n"
            "- Check 'avoid_when' conditions to ensure you're using the right skill\n"
            "- Use 'triggers' keywords to match user intent"
        )

        header = f"## MD Skills\n\n{instructions}\n\n"

        # Group skills by provider
        provider_skills: dict[str, list[dict]] = {}
        standalone_skills: list[dict] = []

        for skill in md_skills[:max_count]:
            provider = skill.get("provider", "")
            if provider:
                provider_skills.setdefault(provider, []).append(skill)
            else:
                standalone_skills.append(skill)

        accumulated = header
        total_count = len(md_skills)
        shown = 0

        # Helper function to format a skill entry
        def format_skill(s: dict) -> str:
            name = s.get("qualified_name") or s.get("name", "unknown")
            desc = s.get("description", "")
            file_path = s.get("file_path", "")
            metadata = s.get("metadata", {})

            # Truncate description
            if len(desc) > desc_max:
                desc = desc[: desc_max - 3] + "..."

            # Shorten home path
            if home_prefix and file_path.startswith(home_prefix):
                file_path = "~" + file_path[len(home_prefix):]

            lines = [
                f"    <skill>",
                f"      <name>{name}</name>",
                f"      <description>{desc}</description>",
                f"      <location>{file_path}</location>",
            ]

            # Add LLM context fields if present
            triggers = metadata.get("triggers", [])
            if triggers and isinstance(triggers, list):
                lines.append(f"      <triggers>{', '.join(triggers)}</triggers>")

            use_when = metadata.get("use_when", [])
            if use_when and isinstance(use_when, list):
                lines.append(f"      <use_when>")
                for condition in use_when[:3]:  # Limit to 3
                    lines.append(f"        - {condition}")
                lines.append(f"      </use_when>")

            avoid_when = metadata.get("avoid_when", [])
            if avoid_when and isinstance(avoid_when, list):
                lines.append(f"      <avoid_when>")
                for condition in avoid_when[:3]:  # Limit to 3
                    lines.append(f"        - {condition}")
                lines.append(f"      </avoid_when>")

            examples = metadata.get("examples", [])
            if examples and isinstance(examples, list):
                lines.append(f"      <examples>")
                for ex in examples[:2]:  # Limit to 2
                    lines.append(f"        - {ex}")
                lines.append(f"      </examples>")

            lines.append(f"    </skill>")
            return "\n".join(lines) + "\n"

        # Helper to format provider context
        def format_provider_context(provider_type: str, ctx: dict) -> str:
            lines = [f"  <provider type=\"{provider_type}\">"]

            display_name = ctx.get("display_name", provider_type)
            if display_name:
                lines.append(f"    <display_name>{display_name}</display_name>")

            description = ctx.get("description", "")
            if description:
                # Truncate description to 200 chars
                if len(description) > 200:
                    description = description[:197] + "..."
                lines.append(f"    <description>{description}</description>")

            keywords = ctx.get("keywords", [])
            if keywords and isinstance(keywords, list):
                lines.append(f"    <keywords>{', '.join(keywords[:10])}</keywords>")

            capabilities = ctx.get("capabilities", [])
            if capabilities and isinstance(capabilities, list):
                lines.append(f"    <capabilities>")
                for cap in capabilities[:5]:  # Limit to 5
                    lines.append(f"      - {cap}")
                lines.append(f"    </capabilities>")

            use_when = ctx.get("use_when", [])
            if use_when and isinstance(use_when, list):
                lines.append(f"    <use_when>")
                for condition in use_when[:3]:
                    lines.append(f"      - {condition}")
                lines.append(f"    </use_when>")

            avoid_when = ctx.get("avoid_when", [])
            if avoid_when and isinstance(avoid_when, list):
                lines.append(f"    <avoid_when>")
                for condition in avoid_when[:3]:
                    lines.append(f"      - {condition}")
                lines.append(f"    </avoid_when>")

            lines.append(f"    <skills>")
            return "\n".join(lines) + "\n"

        # Build provider sections
        accumulated += "<available_skills>\n"

        for provider_type, skills_list in sorted(provider_skills.items()):
            ctx = provider_contexts.get(provider_type, {})

            # Add provider context header
            provider_header = format_provider_context(provider_type, ctx)
            if len(accumulated) + len(provider_header) > budget:
                break
            accumulated += provider_header

            # Add skills under this provider
            for s in skills_list:
                entry = format_skill(s)
                if len(accumulated) + len(entry) + 50 > budget:  # Reserve for closing tags
                    break
                accumulated += entry
                shown += 1

            # Close provider section
            accumulated += "    </skills>\n  </provider>\n"

        # Add standalone skills (no provider)
        if standalone_skills:
            accumulated += "  <standalone_skills>\n"
            for s in standalone_skills:
                entry = format_skill(s)
                if len(accumulated) + len(entry) + 50 > budget:
                    break
                accumulated += entry
                shown += 1
            accumulated += "  </standalone_skills>\n"

        # Add truncation note if needed
        if shown < total_count:
            accumulated += f"  <!-- Showing {shown} of {total_count} skills -->\n"

        accumulated += "</available_skills>"
        return accumulated
    
    def _build_self_update(self) -> str:
        """Build the self-update section."""
        return """## Self-Update

To apply configuration changes, use the appropriate configuration commands."""
    
    def _build_workspace_info(self) -> str:
        """Build the workspace information section."""
        workspace = Path(self.config.workspace_path).expanduser()
        return f"""## Workspace

Working directory: `{workspace}`

You can read and write files in this directory."""
    
    def _build_documentation(self) -> str:
        """Build the documentation section."""
        return """## Documentation

Local documentation path: `docs/`

To understand AtlasClaw's behavior, commands, configuration, or architecture, please refer to the local documentation first."""
    
    def _build_bootstrap(self) -> str:
        """


inject Bootstrap
 
 BOOTSTRAP.md at workspace inject(.atlasclaw_new_workspace).
 (AGENTS.md, SOUL.md etc.) inject.
 
"""
        workspace = Path(self.config.workspace_path).expanduser()
        sections = ["## Project Context", ""]
        
        # workspace(through)
        marker_file = workspace / self.config.new_workspace_marker
        is_new_workspace = marker_file.exists()
        
        any_found = False
        for filename in self.BOOTSTRAP_FILES:
            # BOOTSTRAP.md at workspace inject
            if filename == "BOOTSTRAP.md" and not is_new_workspace:
                continue
                
            filepath = workspace / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    if len(content) > self.config.bootstrap_max_chars:
                        content = (
                            content[:self.config.bootstrap_max_chars]
                            + f"\n...[Truncated at {self.config.bootstrap_max_chars} characters]"
                        )
                    sections.append(f"### {filename}\n\n{content}")
                    any_found = True
                except Exception as e:
                    sections.append(f"### {filename}\n\n[Read failed: {e}]")
        
        # inject BOOTSTRAP.md workspace(run)
        if is_new_workspace and marker_file.exists():
            try:
                marker_file.unlink()
            except Exception:
                pass  # 
        
        return "\n\n".join(sections) if any_found else ""
    
    def is_new_workspace(self) -> bool:
        """

check workspace
        
        Returns:
            such as workspace at, return True
        
"""
        workspace = Path(self.config.workspace_path).expanduser()
        marker_file = workspace / self.config.new_workspace_marker
        return marker_file.exists()
    
    def mark_as_new_workspace(self) -> None:
        """


convertworkspace workspace
 
 create, build inject BOOTSTRAP.md.
 
"""
        workspace = Path(self.config.workspace_path).expanduser()
        workspace.mkdir(parents=True, exist_ok=True)
        marker_file = workspace / self.config.new_workspace_marker
        marker_file.touch(exist_ok=True)
    
    def _build_sandbox(self) -> str:
        """Build the sandbox section."""
        return f"""## Sandbox

Mode: {self.config.sandbox.mode}
Sandbox path: {self.config.sandbox.workspace_root}
Elevated execution: {"Available" if self.config.sandbox.elevated_exec else "Unavailable"}

In sandbox mode, some operations may be restricted."""
    
    def _build_datetime(self) -> str:
        """Build the datetime section."""
        now = datetime.now()
        tz = self.config.user_timezone or "System timezone"
        
        if self.config.time_format == "12":
            time_str = now.strftime("%Y-%m-%d %I:%M:%S %p")
        else:
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""## Current Time

Timezone: {tz}
Current time: {time_str}"""
    
    def _build_reply_tags(self) -> str:
        """build reply"""
        # optional feature, implement
        return ""
    
    def _build_heartbeats(self) -> str:
        """Build the heartbeat section when enabled."""
        # optional feature, implement
        return ""
    
    def _build_runtime_info(self) -> str:
        """Build the runtime information section."""
        return f"""## Runtime

Host: {platform.node()}
OS: {platform.system()} {platform.release()}
Python: {platform.python_version()}
Framework: AtlasClaw v0.1.0"""
    
    def get_context_info(self, detail: bool = False) -> dict:
        """Return prompt context metrics used by the `/context` command."""
        workspace = Path(self.config.workspace_path).expanduser()
        
        files_info = []
        total_size = 0
        
        # workspace
        is_new_workspace = self.is_new_workspace()
        
        for filename in self.BOOTSTRAP_FILES:
            # BOOTSTRAP.md at workspace
            if filename == "BOOTSTRAP.md" and not is_new_workspace:
                files_info.append({
                    "filename": filename,
                    "status": "Skipped (not a new workspace)",
                })
                continue
                
            filepath = workspace / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    raw_size = len(content)
                    injected_size = min(raw_size, self.config.bootstrap_max_chars)
                    truncated = raw_size > self.config.bootstrap_max_chars
                    
                    file_entry = {
                        "filename": filename,
                        "raw_size": raw_size,
                        "injected_size": injected_size,
                        "truncated": truncated,
                    }
                    
                    # mode token estimate(4 characters/token)
                    if detail:
                        file_entry["estimated_tokens"] = injected_size // 4
                        
                    files_info.append(file_entry)
                    total_size += injected_size
                except Exception:
                    files_info.append({
                        "filename": filename,
                        "error": "Read failed",
                    })
            else:
                files_info.append({
                    "filename": filename,
                    "status": "Missing",
                })
        
        result = {
            "mode": self.config.mode.value,
            "workspace": str(workspace),
            "is_new_workspace": is_new_workspace,
            "bootstrap_files": files_info,
            "total_bootstrap_size": total_size,
            "bootstrap_max_chars": self.config.bootstrap_max_chars,
        }
        
        # mode
        if detail:
            result["estimated_bootstrap_tokens"] = total_size // 4
            result["sections"] = self._get_section_breakdown()
            
        return result
    
    def _get_section_breakdown(self) -> list[dict]:
        """

get system prompt
        
        Returns:
            list
        
"""
        sections = []
        
        # estimate
        section_estimates = [
            ("identity", "Identity", len(self._build_identity())),
            ("safety", "Safety", len(self._build_safety())),
            ("self_update", "Self-Update", len(self._build_self_update())),
            ("workspace", "Workspace", len(self._build_workspace_info())),
            ("documentation", "Documentation", len(self._build_documentation())),
            ("datetime", "DateTime", len(self._build_datetime())),
            ("runtime", "Runtime", len(self._build_runtime_info())),
        ]
        
        if self.config.sandbox.enabled:
            section_estimates.append(
                ("sandbox", "Sandbox", len(self._build_sandbox()))
            )
        
        for section_id, section_name, char_count in section_estimates:
            sections.append({
                "id": section_id,
                "name": section_name,
                "char_count": char_count,
                "estimated_tokens": char_count // 4,
            })
            
        return sections
    
    def get_context_detail(
        self,
        skills: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Return detailed prompt context metrics for `/context detail`.

        Includes token overhead estimates for the active skills list and tool
        metadata sections.
        """
        # Start from the base context summary.
        result = self.get_context_info(detail=True)
        
        # Calculate skills-section overhead.
        skills_info = {
            "count": 0,
            "char_count": 0,
            "estimated_tokens": 0,
            "items": [],
        }
        
        if skills:
            skills_listing = self._build_skills_listing(skills)
            skills_info["count"] = len(skills)
            skills_info["char_count"] = len(skills_listing)
            skills_info["estimated_tokens"] = len(skills_listing) // 4
            
            for s in skills:
                skills_info["items"].append({
                    "name": s.get("name", "unknown"),
                    "description_length": len(s.get("description", "")),
                })
        
        result["skills"] = skills_info
        
        # calculate Tools schema overhead
        tools_info = {
            "count": 0,
            "char_count": 0,
            "estimated_tokens": 0,
            "items": [],
        }
        
        if tools:
            tools_listing = self._build_tooling(tools)
            tools_info["count"] = len(tools)
            tools_info["char_count"] = len(tools_listing)
            tools_info["estimated_tokens"] = len(tools_listing) // 4
            
            for t in tools:
                tools_info["items"].append({
                    "name": t.get("name", "unknown"),
                    "description_length": len(t.get("description", "")),
                })
        
        result["tools"] = tools_info
        
        # calculate overhead
        total_chars = (
            result["total_bootstrap_size"]
            + skills_info["char_count"]
            + tools_info["char_count"]
            + sum(s["char_count"] for s in result.get("sections", []))
        )
        
        result["total_system_prompt_chars"] = total_chars
        result["total_estimated_tokens"] = total_chars // 4
        
        return result
