#!/usr/bin/env python
"""
Custom manage-like entry point that

1. Loads variables from config/.env.app (via python-dotenv)
2. Sets a default DJANGO_SETTINGS_MODULE if the .env file hasn’t already
3. Delegates the command to Django’s management CLI
"""

from pathlib import Path
import os
import sys


def load_dotenv() -> None:
    """
    Explicitly load variables from config/.env.app.

    The file is optional; if it’s missing we just continue.
    """
    env_path = Path(__file__).resolve().parent / "config" / ".env.app"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This script needs the python-dotenv package to read config/.env.app. "
            "Install it with `pip install python-dotenv`."
        ) from exc

    # `override=False` keeps any variables already in the parent environment.
    load_dotenv(dotenv_path=env_path, override=False)


def main() -> None:
    load_dotenv()

    # Respect DJANGO_SETTINGS_MODULE from the .env file; fall back otherwise.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")

    try:
        from django.core import management  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    management.execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

