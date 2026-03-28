import os

from flask import Flask, request, jsonify
from visual_memory.utils.logger import setup_crash_handler

from visual_memory.api.routes.health import health_bp
from visual_memory.api.routes.remember import remember_bp
from visual_memory.api.routes.scan import scan_bp
from visual_memory.api.routes.feedback import feedback_bp
from visual_memory.api.routes.retrain import retrain_bp
from visual_memory.api.routes.settings_route import settings_bp
from visual_memory.api.routes.user_settings_route import user_settings_bp
from visual_memory.api.routes.crop import crop_bp
from visual_memory.api.routes.find import find_bp
from visual_memory.api.routes.ask import ask_bp
from visual_memory.api.routes.item_ask import item_ask_bp
from visual_memory.api.routes.items import items_bp
from visual_memory.api.routes.sightings import sightings_bp
from visual_memory.api.routes.debug import debug_bp

_API_KEY = os.environ.get("API_KEY", "")


def create_app():
    setup_crash_handler()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

    @app.before_request
    def _check_api_key():
        if not _API_KEY:
            return
        if request.endpoint in ("health.health", None):
            return
        if request.headers.get("X-API-Key") != _API_KEY:
            return jsonify({"error": "unauthorized"}), 401

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
    app.register_blueprint(debug_bp)

    from visual_memory.api.pipelines import warm_all
    warm_all()

    return app
