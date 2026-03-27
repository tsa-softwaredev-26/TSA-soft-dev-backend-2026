"""Compatibility WSGI shim.

Prefer `services.core.wsgi:application` for new deployments.
"""

from services.core.wsgi import application
