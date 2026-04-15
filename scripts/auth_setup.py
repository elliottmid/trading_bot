#!/usr/bin/env python3
"""
auth_setup.py — Interactive Schwab OAuth setup script.

Walks the user through generating a Schwab API access token for the first
time and verifies it by fetching account information.

Usage:
    python scripts/auth_setup.py [--env /path/to/.env]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.logger import get_logger

log = get_logger(__name__)


_INSTRUCTIONS = """
=============================================================
  Schwab API Authentication Setup
=============================================================

Before running this script, make sure you have:

1.  A Schwab developer account at https://developer.schwab.com/
2.  An application created with the callback URL set to:
        https://127.0.0.1:8182
3.  Your App Key (SCHWAB_API_KEY) and App Secret (SCHWAB_API_SECRET)
    copied into your .env file (see .env.example for the template).

This script will:
  a. Open a browser window for you to log in to Schwab.
  b. After you approve access, Schwab will redirect you to a URL that
     starts with https://127.0.0.1:8182/?code=...
  c. Paste that full redirect URL back here when prompted.
  d. The script will exchange the code for tokens and save them to
     the path configured in SCHWAB_TOKEN_PATH.

=============================================================
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schwab OAuth token setup.")
    parser.add_argument("--env", default=None, help="Path to .env file.")
    return parser.parse_args()


def main() -> None:
    """Run the interactive auth setup flow."""
    args = parse_args()
    config = Config.from_env(dotenv_path=args.env)

    print(_INSTRUCTIONS)

    if not config.schwab_api_key:
        print(
            "ERROR: SCHWAB_API_KEY is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
        sys.exit(1)

    if not config.schwab_api_secret:
        print(
            "ERROR: SCHWAB_API_SECRET is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
        sys.exit(1)

    try:
        import schwab
    except ImportError:
        print("ERROR: schwab-py is not installed. Run: pip install schwab-py")
        sys.exit(1)

    token_path = config.schwab_token_path
    Path(token_path).parent.mkdir(parents=True, exist_ok=True)

    print("Starting OAuth flow...")
    print("A browser window will open. Log in, approve access, then paste")
    print("the redirect URL back here.\n")

    try:
        client = schwab.auth.client_from_login_flow(
            api_key=config.schwab_api_key,
            app_secret=config.schwab_api_secret,
            callback_url=config.schwab_callback_url,
            token_path=token_path,
        )
    except Exception as exc:
        print("\nERROR during OAuth flow: %s" % exc)
        print(
            "Make sure your App Key, Secret, and callback URL are correct "
            "and that the callback URL is registered in the developer portal."
        )
        sys.exit(1)

    print("\nToken saved to: %s" % token_path)
    print("\nVerifying token by fetching account numbers...")

    try:
        resp = client.get_account_numbers()
        resp.raise_for_status()
        accounts = resp.json()
        if accounts:
            print("SUCCESS — found %d account(s)." % len(accounts))
            for acct in accounts:
                print(
                    "  Account (last 4): ...%s"
                    % acct.get("accountNumber", "????")[-4:]
                )
        else:
            print("WARNING: authenticated but no accounts found.")
    except Exception as exc:
        print("WARNING: token saved but account fetch failed: %s" % exc)

    print("\nSetup complete.  Run 'python scripts/run_bot.py' to start the bot.")


if __name__ == "__main__":
    main()
