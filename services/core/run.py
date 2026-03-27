"""Run the core backend in development mode."""

from visual_memory.api.app import create_app
from visual_memory.config import Settings


def main() -> None:
    settings = Settings()
    app = create_app()
    app.run(host=settings.api_host, port=settings.api_port, threaded=False)


if __name__ == "__main__":
    main()
