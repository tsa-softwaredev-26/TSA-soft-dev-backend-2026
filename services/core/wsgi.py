"""WSGI entrypoint for the core backend."""

from visual_memory.api.app import create_app

application = create_app()
