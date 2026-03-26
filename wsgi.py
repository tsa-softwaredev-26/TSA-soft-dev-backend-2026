"""WSGI entry point for gunicorn.

Usage (production, Debian server):
    gunicorn -w 1 -b 127.0.0.1:5000 wsgi:application

Single worker required: model weights are held in process memory and are not
safe to share across workers without a separate model server.
"""
from visual_memory.api.app import create_app

application = create_app()
