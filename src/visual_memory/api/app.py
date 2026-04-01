import os
import sys

from flask import Flask, jsonify, request
from flask_socketio import SocketIO

from visual_memory.api.routes.ask import ask_bp
from visual_memory.api.routes.crop import crop_bp
from visual_memory.api.routes.debug import debug_bp
from visual_memory.api.routes.feedback import feedback_bp
from visual_memory.api.routes.find import find_bp
from visual_memory.api.routes.health import health_bp
from visual_memory.api.routes.item_ask import item_ask_bp
from visual_memory.api.routes.items import items_bp
from visual_memory.api.routes.remember import remember_bp
from visual_memory.api.routes.retrain import retrain_bp
from visual_memory.api.routes.scan import scan_bp
from visual_memory.api.routes.settings_route import settings_bp
from visual_memory.api.routes.sightings import sightings_bp
from visual_memory.api.routes.transcribe import transcribe_bp
from visual_memory.api.routes.user_settings_route import user_settings_bp
from visual_memory.api.routes.voice import voice_bp
from visual_memory.utils.logger import get_logger, setup_crash_handler
from visual_memory.utils.memory_monitor import MemoryMonitor

_API_KEY = os.environ.get("API_KEY", "")
_logger = get_logger(__name__)
_memory_monitor = MemoryMonitor()

socketio = SocketIO()


def _validate_api_key() -> None:
    if not _API_KEY:
        _logger.warning({"event": "api_key_unset", "message": "API auth disabled"})
        return
    if len(_API_KEY) < 32:
        _logger.error({"event": "api_key_weak", "reason": "too_short", "length": len(_API_KEY)})
        raise SystemExit("API_KEY must be at least 32 characters. Generate with: openssl rand -hex 32")
    lowered = _API_KEY.lower()
    _logger.info({"event": "api_key_valid", "length": len(_API_KEY)})


def create_app():
    setup_crash_handler()
    try:
        _validate_api_key()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        raise

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    @app.before_request
    def _check_memory_pressure():
        if _memory_monitor.is_oom_risk(threshold=0.90):
            _memory_monitor.log_memory_state(level="critical")
            return jsonify({
                "error": "server_overloaded",
                "message": "Memory exhausted, try again later",
            }), 503
        if _memory_monitor.suggest_throttle():
            _memory_monitor.log_memory_state(level="warning")

    @app.before_request
    def _check_api_key():
        if not _API_KEY:
            return
        if request.endpoint in ("health.health", None):
            return
        if request.headers.get("X-API-Key") != _API_KEY:
            return jsonify({"error": "unauthorized"}), 401

    socketio.init_app(
        app,
        async_mode="gevent",
        cors_allowed_origins="*",
        max_http_buffer_size=100 * 1024 * 1024,
        logger=False,
        engineio_logger=False,
    )

    from visual_memory.api.routes.voice_ws import register_events

    register_events(socketio)

    app.register_blueprint(health_bp)
    app.register_blueprint(remember_bp)
    app.register_blueprint(scan_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(retrain_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(user_settings_bp)
    app.register_blueprint(crop_bp)
    app.register_blueprint(find_bp)
    app.register_blueprint(ask_bp)
    app.register_blueprint(item_ask_bp)
    app.register_blueprint(items_bp)
    app.register_blueprint(sightings_bp)
    if os.environ.get("ENABLE_DEBUG_ROUTES", "0") == "1":
        app.register_blueprint(debug_bp)
    app.register_blueprint(transcribe_bp)
    app.register_blueprint(voice_bp)

    from visual_memory.api.pipelines import warm_all

    warm_all()
    return app
