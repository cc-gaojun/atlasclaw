# -*- coding: utf-8 -*-
"""
JIRA Provider E2E tests.

Live integration tests against real JIRA Server/DC instance.
Run:
    python -m pytest tests/atlasclaw/providers/test_jira_e2e.py -v -s

Config is read from ATLASCLAW_CONFIG (default: tests/atlasclaw.test.json) -> service_providers.jira -> first instance.

For Agent E2E tests, the service must be running:
    uvicorn app.atlasclaw.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent.parent

load_dotenv(dotenv_path=_ROOT / ".env", override=False)

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import httpx
import pytest

sys.path.insert(0, str(_ROOT))

pytestmark = [pytest.mark.e2e]

_ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

def _resolve_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    m = _ENV_REF_PATTERN.match(value.strip())
    if not m:
        return value
    env_name = m.group(1)
    return os.environ.get(env_name, "")



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jira_config() -> dict:
    """Load JIRA connection config from ATLASCLAW_CONFIG (default tests/atlasclaw.test.json)."""
    config_path = Path(
        os.environ.get("ATLASCLAW_CONFIG", str(_ROOT / "tests" / "atlasclaw.test.json"))
    ).resolve()
    assert config_path.exists(), f"Config file not found: {config_path}"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    sp = config.get("service_providers", {}).get("jira", {})
    assert sp, "No jira entry in service_providers"

    instance_name = next(iter(sp))
    instance = sp[instance_name]
    instance = {k: _resolve_env(v) for k, v in instance.items()}

    for key in ("base_url", "username"):
        assert instance.get(key), f"Missing required field: jira.{instance_name}.{key}"
    assert instance.get("password") or instance.get("token"), (
        f"Missing required field: jira.{instance_name}.password"
    )

    return instance


@pytest.fixture(scope="module")
def api_version(jira_config: dict) -> str:
    return jira_config.get("api_version", "2")


@pytest.fixture(scope="module")
def jira_client(jira_config: dict, api_version: str):
    """Create a synchronous httpx client for JIRA REST API."""
    base_url = jira_config["base_url"].rstrip("/")

    with httpx.Client(
        base_url=base_url,
        auth=(jira_config["username"], jira_config.get("password", jira_config.get("token", ""))),
        headers={"Content-Type": "application/json"},
        timeout=30.0,
        proxy=None,
        trust_env=False,
    ) as client:
        resp = client.get(f"/rest/api/{api_version}/serverInfo")
        assert resp.status_code == 200, (
            f"Cannot connect to JIRA at {base_url}: {resp.status_code} {resp.text[:200]}"
        )
        yield client


@pytest.fixture(scope="module")
def project_key(jira_client: httpx.Client, api_version: str) -> str:
    """Auto-detect a usable project key (prefer one with existing issues)."""
    resp = jira_client.get(f"/rest/api/{api_version}/project")
    assert resp.status_code == 200, f"Cannot list projects: {resp.status_code}"
    projects = resp.json()
    assert projects, "No JIRA projects available on this server"

    for proj in projects:
        key = proj["key"]
        search = jira_client.get(
            f"/rest/api/{api_version}/search",
            params={"jql": f"project = {key}", "maxResults": 1, "fields": "key"},
        )
        if search.status_code == 200 and search.json().get("total", 0) > 0:
            return key

    return projects[0]["key"]


@pytest.fixture(scope="module")
def created_issue_key(jira_client: httpx.Client, project_key: str, api_version: str):
    """
    Create a test issue and yield its key.
    Shared across tests that need an existing issue.
    Cleanup after all tests in this module.
    """
    meta_resp = jira_client.get(
        f"/rest/api/{api_version}/issue/createmeta",
        params={"projectKeys": project_key, "expand": "projects.issuetypes.fields"},
    )

    fields_payload: dict = {
        "project": {"key": project_key},
        "summary": "[E2E Test] AtlasClaw automated test issue",
        "description": "Created by test_jira_e2e.py. Safe to delete.",
        "issuetype": {"name": "Task"},
    }

    if meta_resp.status_code == 200:
        meta = meta_resp.json()
        for proj in meta.get("projects", []):
            if proj["key"] != project_key:
                continue
            for itype in proj.get("issuetypes", []):
                if itype["name"] != "Task":
                    continue
                for field_key, field_def in itype.get("fields", {}).items():
                    if not field_def.get("required"):
                        continue
                    if field_key in ("project", "summary", "issuetype", "description", "reporter"):
                        continue
                    if field_key == "components":
                        comp_resp = jira_client.get(
                            f"/rest/api/{api_version}/project/{project_key}/components"
                        )
                        if comp_resp.status_code == 200:
                            comps = comp_resp.json()
                            if comps:
                                fields_payload["components"] = [{"id": comps[0]["id"]}]
                    elif field_key == "priority":
                        fields_payload.setdefault("priority", {"name": "Medium"})

    resp = jira_client.post(
        f"/rest/api/{api_version}/issue",
        json={"fields": fields_payload},
    )
    assert resp.status_code in (200, 201), (
        f"Setup: create issue failed: {resp.status_code} {resp.text[:500]}"
    )

    issue_key = resp.json()["key"]
    yield issue_key

    try:
        jira_client.delete(f"/rest/api/{api_version}/issue/{issue_key}")
    except Exception:
        pass


@pytest.fixture(scope="module")
def api_base_url() -> str:
    """AtlasClaw API base URL."""
    return os.environ.get("ATLASCLAW_API_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="module")
def api_client(api_base_url: str):
    """HTTP client for AtlasClaw API."""
    with httpx.Client(
        base_url=api_base_url,
        timeout=120.0,
        proxy=None,
        trust_env=False,
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------


class TestJiraE2E:
    """Live E2E tests against real JIRA Server/DC."""

    def test_list_active_issues(
        self, jira_client: httpx.Client, project_key: str, api_version: str,
        created_issue_key: str,
    ):
        """Scenario 1: List all active issues in a project."""
        jql = f"project = {project_key} ORDER BY created DESC"
        resp = jira_client.get(
            f"/rest/api/{api_version}/search",
            params={"jql": jql, "maxResults": 50, "fields": "key,summary,status,assignee"},
        )
        assert resp.status_code == 200, f"Search failed: {resp.status_code} {resp.text[:300]}"

        data = resp.json()
        assert "issues" in data
        assert "total" in data
        assert data["total"] > 0, "Expected at least 1 issue (the one we created)"

        keys = [i["key"] for i in data["issues"]]
        assert created_issue_key in keys, (
            f"Created issue {created_issue_key} not found in search results"
        )

        print(f"\n  Project: {project_key}")
        print(f"  Total issues: {data['total']}")
        for issue in data["issues"][:5]:
            fields = issue["fields"]
            status = fields["status"]["name"]
            assignee = (fields.get("assignee") or {}).get("displayName", "Unassigned")
            print(f"    {issue['key']}  [{status}]  {fields['summary'][:60]}  -> {assignee}")

    def test_get_issue_detail(
        self, jira_client: httpx.Client, api_version: str,
        created_issue_key: str,
    ):
        """Scenario 2: Get a specific issue's full details."""
        resp = jira_client.get(f"/rest/api/{api_version}/issue/{created_issue_key}")
        assert resp.status_code == 200, (
            f"Get issue failed: {resp.status_code} {resp.text[:300]}"
        )

        issue = resp.json()
        fields = issue["fields"]

        assert issue["key"] == created_issue_key
        assert "summary" in fields
        assert "status" in fields
        assert "issuetype" in fields
        assert fields["summary"].startswith("[E2E Test]")

        print(f"\n  Issue:       {issue['key']}")
        print(f"  Summary:     {fields['summary']}")
        print(f"  Type:        {fields['issuetype']['name']}")
        print(f"  Status:      {fields['status']['name']}")
        print(f"  Priority:    {fields.get('priority', {}).get('name', 'N/A')}")
        assignee = (fields.get("assignee") or {}).get("displayName", "Unassigned")
        reporter = (fields.get("reporter") or {}).get("displayName", "N/A")
        print(f"  Assignee:    {assignee}")
        print(f"  Reporter:    {reporter}")
        print(f"  Created:     {fields.get('created', 'N/A')}")
        print(f"  Updated:     {fields.get('updated', 'N/A')}")


    def test_update_issue(
        self, jira_client: httpx.Client, api_version: str,
        created_issue_key: str,
    ):
        """Scenario 4: Update an existing issue and verify field changes."""
        updated_summary = "[E2E Test] Updated summary from CRUD verification"
        updated_desc = "Updated by test_jira_e2e.py test_update_issue."

        resp = jira_client.put(
            f"/rest/api/{api_version}/issue/{created_issue_key}",
            json={
                "fields": {
                    "summary": updated_summary,
                    "description": updated_desc,
                }
            },
        )
        assert resp.status_code in (200, 204), (
            f"Update issue failed: {resp.status_code} {resp.text[:300]}"
        )

        verify_resp = jira_client.get(f"/rest/api/{api_version}/issue/{created_issue_key}")
        assert verify_resp.status_code == 200, (
            f"Verify updated issue failed: {verify_resp.status_code} {verify_resp.text[:300]}"
        )

        fields = verify_resp.json()["fields"]
        assert fields.get("summary") == updated_summary

        desc_value = fields.get("description")
        if isinstance(desc_value, str):
            assert "Updated by test_jira_e2e.py test_update_issue." in desc_value
        elif isinstance(desc_value, dict):
            desc_text = str(desc_value)
            assert "Updated by test_jira_e2e.py test_update_issue." in desc_text

        print(f"\n  Updated:  {created_issue_key}")
        print(f"  Summary:  {fields.get('summary')}")

    def test_create_issue(
        self, jira_client: httpx.Client, project_key: str, api_version: str,
    ):
        """Scenario 3: Create a new issue and verify it exists, then delete."""
        fields_payload: dict = {
            "project": {"key": project_key},
            "summary": "[E2E Test] Create-and-delete verification",
            "description": "Created by test_jira_e2e.py test_create_issue. Will be deleted.",
            "issuetype": {"name": "Task"},
        }

        meta_resp = jira_client.get(
            f"/rest/api/{api_version}/issue/createmeta",
            params={"projectKeys": project_key, "expand": "projects.issuetypes.fields"},
        )
        if meta_resp.status_code == 200:
            meta = meta_resp.json()
            for proj in meta.get("projects", []):
                if proj["key"] != project_key:
                    continue
                for itype in proj.get("issuetypes", []):
                    if itype["name"] != "Task":
                        continue
                    for field_key, field_def in itype.get("fields", {}).items():
                        if not field_def.get("required"):
                            continue
                        if field_key in ("project", "summary", "issuetype", "description", "reporter"):
                            continue
                        if field_key == "components":
                            comp_resp = jira_client.get(
                                f"/rest/api/{api_version}/project/{project_key}/components"
                            )
                            if comp_resp.status_code == 200:
                                comps = comp_resp.json()
                                if comps:
                                    fields_payload["components"] = [{"id": comps[0]["id"]}]
                        elif field_key == "priority":
                            fields_payload.setdefault("priority", {"name": "Medium"})

        resp = jira_client.post(
            f"/rest/api/{api_version}/issue",
            json={"fields": fields_payload},
        )
        assert resp.status_code in (200, 201), (
            f"Create issue failed: {resp.status_code} {resp.text[:500]}"
        )

        created = resp.json()
        assert "key" in created
        assert "id" in created
        issue_key = created["key"]

        print(f"\n  Created:  {issue_key}")
        print(f"  ID:       {created['id']}")

        verify_resp = jira_client.get(f"/rest/api/{api_version}/issue/{issue_key}")
        assert verify_resp.status_code == 200, f"Created issue not found: {issue_key}"
        assert verify_resp.json()["fields"]["summary"].startswith("[E2E Test]")
        print(f"  Verified: {issue_key} exists with correct summary")

        del_resp = jira_client.delete(f"/rest/api/{api_version}/issue/{issue_key}")
        if del_resp.status_code in (200, 204):
            print(f"  Cleanup:  deleted {issue_key}")
        else:
            print(f"  Cleanup:  delete returned {del_resp.status_code} (manual cleanup needed)")




class TestJiraAgentE2E:
    """
    Full Agent E2E tests via HTTP API.
    
    Prerequisites:
    1. Service must be running with atlasclaw.test.json config:
       $env:ATLASCLAW_CONFIG="atlasclaw.test.json"
       uvicorn app.atlasclaw.main:app --host 0.0.0.0 --port 8000
    
    2. Jira provider must be configured in atlasclaw.test.json
    """

    @pytest.mark.llm
    def test_health_check(self, api_client: httpx.Client):
        """Verify the service is running."""
        resp = api_client.get("/api/health")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        data = resp.json()
        assert data.get("status") == "healthy"
        print(f"\n  Service health: {data['status']}")

    @pytest.mark.llm
    def test_agent_create_jira_issue_via_api(
        self,
        api_client: httpx.Client,
        jira_config: dict,
        jira_client: httpx.Client,
        api_version: str,
    ):
        """
        Complete E2E test via HTTP API:
        1. Create session via POST /api/sessions
        2. Send user message via POST /api/agent/run
        3. Stream response via GET /api/agent/runs/{run_id}/stream
        4. Verify Jira issue was created
        5. Cleanup
        """
        session_resp = api_client.post(
            "/api/sessions",
            json={
                "agent_id": "main",
                "channel": "e2e-test",
                "chat_type": "dm",
                "scope": "main",
            },
        )
        assert session_resp.status_code == 200, f"Create session failed: {session_resp.status_code} {session_resp.text}"
        session_data = session_resp.json()
        session_key = session_data["session_key"]
        print(f"\n  Created session: {session_key}")

        user_prompt = "Create a Jira Task issue in project TEST. Summary: Service startup failure. Description: Service failed to start due to a database connection timeout."

        run_resp = api_client.post(
            "/api/agent/run",
            json={
                "session_key": session_key,
                "message": user_prompt,
                "timeout_seconds": 120,
            },
        )
        assert run_resp.status_code == 200, f"Start agent run failed: {run_resp.status_code} {run_resp.text}"
        run_data = run_resp.json()
        run_id = run_data["run_id"]
        print(f"  Started agent run: {run_id}")

        response_content = []
        current_event_type = None
        with api_client.stream("GET", f"/api/agent/runs/{run_id}/stream") as stream:
            for line in stream.iter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event_type = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        event_data = json.loads(data_str)
                        if current_event_type == "assistant":
                            text = event_data.get("text", "")
                            response_content.append(text)
                            print(f"  [assistant] {text[:100]}...")
                        elif current_event_type == "tool":
                            tool_name = event_data.get("tool", "")
                            phase = event_data.get("phase", "")
                            print(f"  [tool] {tool_name} - {phase}")
                        elif current_event_type == "lifecycle":
                            phase = event_data.get("phase", "")
                            print(f"  [lifecycle] {phase}")
                            if phase == "end" or phase == "error":
                                break
                        elif current_event_type == "error":
                            error = event_data.get("message", "")
                            print(f"  [error] {error}")
                            break
                    except json.JSONDecodeError:
                        pass

        full_response = "".join(response_content)
        print(f"\n  Full response: {full_response[:500]}")

        issue_key_pattern = r"[A-Z]+-\d+"
        match = re.search(issue_key_pattern, full_response)
        assert match, f"Agent response should contain a Jira issue key, got: {full_response[:300]}"
        
        issue_key = match.group(0)
        print(f"  Extracted issue key: {issue_key}")

        verify_resp = jira_client.get(f"/rest/api/{api_version}/issue/{issue_key}")
        assert verify_resp.status_code == 200, f"Created issue not found in Jira: {issue_key}"
        
        issue_data = verify_resp.json()
        summary = issue_data["fields"]["summary"]
        print(f"  Verified: {issue_key} exists in Jira with summary: {summary}")

        try:
            jira_client.delete(f"/rest/api/{api_version}/issue/{issue_key}")
            print(f"  Cleanup: deleted {issue_key}")
        except Exception as e:
            print(f"  Cleanup warning: {e}")
