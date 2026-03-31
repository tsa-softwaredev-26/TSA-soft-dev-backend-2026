"""Run the core backend in development mode."""

from visual_memory.api.app import create_app, socketio
from visual_memory.config import Settings


def main() -> None:
    settings = Settings()
    app = create_app()
    socketio.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
