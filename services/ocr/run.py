"""Run the OCR service in development mode."""

import os

import uvicorn


def main() -> None:
    host = os.environ.get("OCR_HOST", "127.0.0.1")
    port = int(os.environ.get("OCR_PORT", "8001"))
    uvicorn.run("services.ocr.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
