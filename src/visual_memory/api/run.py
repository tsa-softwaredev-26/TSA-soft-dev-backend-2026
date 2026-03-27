"""Compatibility wrapper for the core backend dev entrypoint.

Prefer `python -m services.core.run` or `python services/core/run.py`.
"""

from services.core.run import main


if __name__ == "__main__":
    main()
