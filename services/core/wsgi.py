"""WSGI entrypoint for the core backend.

Gunicorn command (gevent worker required for WebSocket support):
    gunicorn -w 1 -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
        -b 127.0.0.1:5000 --timeout 0 services.core.wsgi:application
"""

from visual_memory.api.app import create_app

application = create_app()
