# -*- coding: utf-8 -*-
"""DingTalk OAuth2 SSO login flow (non-standard, not OIDC)."""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.atlasclaw.auth.models import AuthResult, AuthenticationError
from app.atlasclaw.auth.providers.oidc_sso import OIDCSSOProvider

logger = logging.getLogger(__name__)

# DingTalk OAuth2 endpoints (non-standard OIDC)
DINGTALK_AUTHORIZATION_ENDPOINT = "https://login.dingtalk.com/oauth2/auth"
DINGTALK_TOKEN_ENDPOINT = "https://api.dingtalk.com/v1.0/oauth2/userAccessToken"
DINGTALK_USERINFO_ENDPOINT = "https://api.dingtalk.com/v1.0/contact/users/me"


class DingTalkSSOProvider(OIDCSSOProvider):
    """
    DingTalk OAuth2 SSO login flow.

    DingTalk uses a custom OAuth2 flow with the following differences from standard OIDC:
      - Token endpoint uses JSON request body + camelCase parameters (not form-encoded)
      - No id_token is returned, only accessToken + refreshToken
      - Userinfo endpoint uses custom header `x-acs-dingtalk-access-token` (not Bearer)
      - User identifier is obtained from userinfo `unionId`/`openId` (not id_token `sub`)
      - No OIDC Discovery endpoint

    Attributes:
        corp_id: DingTalk enterprise corpId, used as tenant_id mapping.
    """

    def __init__(
        self,
        issuer: str,
        client_id: str,
        client_secret: str = "",
        redirect_uri: str = "",
        authorization_endpoint: str = "",
        token_endpoint: str = "",
        userinfo_endpoint: str = "",
        scopes: Optional[list[str]] = None,
        pkce_enabled: bool = True,
        pkce_method: str = "S256",
        corp_id: str = "",
    ) -> None:
        """
        Initialize DingTalk SSO Provider.

        Args:
            issuer: Issuer URL (for base class compatibility, not used by DingTalk).
            client_id: DingTalk application AppKey.
            client_secret: DingTalk application AppSecret.
            redirect_uri: OAuth2 callback URI.
            authorization_endpoint: Authorization endpoint, defaults to DingTalk official endpoint.
            token_endpoint: Token exchange endpoint, defaults to DingTalk official endpoint.
            userinfo_endpoint: Userinfo endpoint, defaults to DingTalk official endpoint.
            scopes: OAuth2 scopes, defaults to ["openid", "corpid"].
            pkce_enabled: Whether to enable PKCE, defaults to True.
            pkce_method: PKCE method, defaults to "S256".
            corp_id: DingTalk enterprise corpId, used as tenant_id mapping.
        """
        # Use DingTalk default endpoints if not explicitly configured
        super().__init__(
            issuer=issuer,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            authorization_endpoint=authorization_endpoint or DINGTALK_AUTHORIZATION_ENDPOINT,
            token_endpoint=token_endpoint or DINGTALK_TOKEN_ENDPOINT,
            userinfo_endpoint=userinfo_endpoint or DINGTALK_USERINFO_ENDPOINT,
            jwks_uri="",  # DingTalk does not use JWKS
            scopes=scopes or ["openid", "corpid"],
            pkce_enabled=pkce_enabled,
            pkce_method=pkce_method,
        )
        self._corp_id = corp_id

    def build_authorization_url(self, state: str, code_challenge: str = "") -> str:
        """
        Build DingTalk authorization URL with required prompt=consent.

        DingTalk requires prompt=consent parameter in the authorization URL,
        so users will see the authorization confirmation page and explicitly
        consent to permissions like Contact.User.Read.
        Otherwise, the token will not include these permissions, causing userinfo API to return 403.

        See: https://open-dingtalk.github.io/developerpedia/docs/develop/permission/token/browser/get_user_app_token_browser/
        """
        base_url = super().build_authorization_url(state, code_challenge)
        # DingTalk requires prompt=consent to obtain user authorization scope
        return f"{base_url}&prompt=consent"

    async def exchange_code(self, code: str, code_verifier: str = "") -> dict[str, Any]:
        """
        Exchange authorization code for tokens (DingTalk custom format).

        DingTalk uses JSON request body instead of form-encoded:
        - Parameters use camelCase: clientId, clientSecret, code, grantType
        - Does not use HTTP Basic Auth

        Args:
            code: Authorization code.
            code_verifier: PKCE code_verifier (if enabled).

        Returns:
            Token response JSON containing accessToken, refreshToken, expireIn.

        Raises:
            AuthenticationError: When token exchange fails.
        """
        # DingTalk uses JSON request body with camelCase parameters
        payload: dict[str, str] = {
            "clientId": self._client_id,
            "clientSecret": self._client_secret,
            "code": code,
            "grantType": "authorization_code",
        }

        logger.info(
            "[DingTalk SSO] Exchanging code at %s",
            self._token_endpoint,
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    self._token_endpoint,
                    json=payload,  # JSON request body, not form data
                    headers={"Content-Type": "application/json"},
                )
                logger.debug(
                    "[DingTalk SSO] Token exchange response: status=%s body=%s",
                    resp.status_code,
                    resp.text[:500] if resp.text else "",
                )
                resp.raise_for_status()
                token_data = resp.json()
                # Log first 10 chars of accessToken for debugging (not full token)
                at = token_data.get("accessToken", "")
                logger.info(
                    "[DingTalk SSO] Token exchange success, accessToken prefix: %s...",
                    at[:10] if at else "<empty>",
                )
                return token_data
        except httpx.HTTPStatusError as exc:
            logger.error(
                "[DingTalk SSO] Token exchange failed: status=%s body=%s",
                exc.response.status_code,
                exc.response.text[:500] if exc.response.text else "",
            )
            raise AuthenticationError(
                f"DingTalk token exchange failed: {exc.response.status_code}"
            )
        except Exception as exc:
            logger.error("[DingTalk SSO] Token exchange error: %s", exc)
            raise AuthenticationError(f"DingTalk token exchange failed: {exc}")

    async def fetch_userinfo(self, access_token: str) -> dict[str, Any]:
        """
        Fetch user info from DingTalk (custom header format).

        DingTalk uses custom header `x-acs-dingtalk-access-token`, not standard Bearer.

        Args:
            access_token: DingTalk access token.

        Returns:
            Userinfo JSON containing nick, unionId, openId, mobile, email, avatarUrl, etc.

        Raises:
            AuthenticationError: When fetching userinfo fails (network error or API error).
        """
        logger.info(
            "[DingTalk SSO] Fetching userinfo from %s",
            self._userinfo_endpoint,
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    self._userinfo_endpoint,
                    headers={
                        # DingTalk uses custom header, not Authorization: Bearer
                        "x-acs-dingtalk-access-token": access_token,
                    },
                )
                logger.debug(
                    "[DingTalk SSO] Userinfo response: status=%s body=%s",
                    resp.status_code,
                    resp.text[:500] if resp.text else "",
                )
                resp.raise_for_status()
                data = resp.json()

                # DingTalk API may return HTTP 200 but body contains error code
                if "errcode" in data and data["errcode"] != 0:
                    errcode = data.get("errcode")
                    errmsg = data.get("errmsg", "Unknown error")
                    logger.error(
                        "[DingTalk SSO] Userinfo API error: errcode=%s errmsg=%s",
                        errcode,
                        errmsg,
                    )
                    raise AuthenticationError(
                        f"DingTalk userinfo API error: [{errcode}] {errmsg}"
                    )

                return data

        except httpx.HTTPStatusError as exc:
            # HTTP error status code, log response body for debugging
            logger.error(
                "[DingTalk SSO] Userinfo HTTP error: status=%s body=%s",
                exc.response.status_code,
                exc.response.text[:500] if exc.response.text else "",
            )
            raise AuthenticationError(
                f"DingTalk userinfo request failed: HTTP {exc.response.status_code}"
            ) from exc

        except httpx.RequestError as exc:
            # Network/connection error
            logger.error("[DingTalk SSO] Userinfo request error: %s", exc)
            raise AuthenticationError(
                f"DingTalk userinfo request failed: {exc}"
            ) from exc

    async def complete_login(self, code: str, code_verifier: str = "") -> AuthResult:
        """
        Complete DingTalk SSO login flow.

        Differences from standard OIDC:
        - Token response uses camelCase: accessToken (not access_token)
        - No id_token, skips JWT verification
        - User identifier obtained from userinfo (unionId/openId)

        Args:
            code: Authorization code.
            code_verifier: PKCE code_verifier (if enabled).

        Returns:
            AuthResult containing user identity information.

        Raises:
            AuthenticationError: When login flow fails.
        """
        # 1. Exchange authorization code for token
        tokens = await self.exchange_code(code, code_verifier)

        # DingTalk uses camelCase: accessToken (not access_token)
        access_token = tokens.get("accessToken", "")
        if not access_token:
            logger.error(
                "[DingTalk SSO] No accessToken in response: %s",
                list(tokens.keys()),
            )
            raise AuthenticationError("No accessToken in DingTalk token response")

        # 2. Fetch user info (fetch_userinfo raises AuthenticationError on failure)
        userinfo = await self.fetch_userinfo(access_token)

        # 3. Extract user identifier from userinfo
        # Prefer unionId (cross-app unique), fallback to openId (app-local unique)
        subject = userinfo.get("unionId") or userinfo.get("openId", "")
        if not subject:
            logger.error(
                "[DingTalk SSO] No unionId/openId in userinfo: %s",
                list(userinfo.keys()),
            )
            raise AuthenticationError(
                "Missing unionId/openId in DingTalk userinfo response"
            )

        # 4. Build AuthResult
        result = AuthResult(
            subject=subject,
            display_name=userinfo.get("nick", ""),
            email=userinfo.get("email", ""),
            roles=[],
            tenant_id=self._corp_id if self._corp_id else "default",
            raw_token=access_token,
            id_token="",  # DingTalk does not return id_token
            extra={
                "unionId": userinfo.get("unionId", ""),
                "openId": userinfo.get("openId", ""),
                "mobile": userinfo.get("mobile", ""),
                "avatarUrl": userinfo.get("avatarUrl", ""),
                "stateCode": userinfo.get("stateCode", ""),
            },
        )

        logger.info(
            "[DingTalk SSO] Login completed: subject=%s display_name=%s tenant_id=%s",
            result.subject,
            result.display_name,
            result.tenant_id,
        )

        return result
