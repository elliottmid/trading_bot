"""
auth.py — Schwab OAuth token management.

Wraps schwab-py's authentication helpers and provides token-age checks
so stale tokens are caught before the trading loop starts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import schwab

from ..config import Config
from ..logger import get_logger

log = get_logger(__name__)

_TOKEN_WARN_DAYS = 5  # Warn if token is older than this many days


class SchwabAuth:
    """Manage Schwab API authentication tokens.

    Args:
        config: Application configuration instance.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client: Optional[schwab.client.Client] = None

    def get_client(self) -> schwab.client.Client:
        """Return a ready-to-use schwab-py client.

        Attempts to load the token from disk.  If the token file is missing
        or corrupt, logs clear instructions for the user to re-authenticate
        manually.

        Returns:
            Authenticated schwab-py Client.

        Raises:
            FileNotFoundError: If the token file does not exist.
            RuntimeError: If the token cannot be loaded for any other reason.
        """
        if self._client is not None:
            return self._client

        token_path = self._config.schwab_token_path
        if not Path(token_path).exists():
            log.error(
                "Token file not found at '%s'. "
                "Run 'python scripts/auth_setup.py' to generate a token.",
                token_path,
            )
            raise FileNotFoundError(
                "Schwab token not found at %s" % token_path
            )

        try:
            self._client = schwab.auth.client_from_token_file(
                token_path=token_path,
                api_key=self._config.schwab_api_key,
                app_secret=self._config.schwab_api_secret,
            )
            log.info("Schwab client loaded from token file: %s", token_path)
        except Exception as exc:
            log.error(
                "Failed to load Schwab client from token file: %s. "
                "Re-run 'python scripts/auth_setup.py'.",
                exc,
            )
            raise RuntimeError("Could not load Schwab token: %s" % exc) from exc

        self.warn_if_expiring_soon()
        return self._client

    def check_token_age(self) -> int:
        """Return the number of days since the token file was last modified.

        Returns:
            Age of the token file in whole days, or -1 if the file does not
            exist.
        """
        token_path = Path(self._config.schwab_token_path)
        if not token_path.exists():
            return -1
        mtime = token_path.stat().st_mtime
        import time
        age_seconds = time.time() - mtime
        age_days = int(age_seconds / 86400)
        return age_days

    def warn_if_expiring_soon(self) -> None:
        """Log a WARNING if the token file is older than ``_TOKEN_WARN_DAYS`` days.

        Schwab refresh tokens expire after 7 days of inactivity.  This check
        gives the user advance notice to re-authenticate.
        """
        age = self.check_token_age()
        if age < 0:
            log.warning("Token file not found; cannot check age.")
        elif age >= _TOKEN_WARN_DAYS:
            log.warning(
                "Schwab token is %d days old (warning threshold: %d days). "
                "Consider running 'python scripts/auth_setup.py' to refresh.",
                age,
                _TOKEN_WARN_DAYS,
            )
        else:
            log.info("Token age: %d day(s) — OK.", age)
