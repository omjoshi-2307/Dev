"""DEV API application entry point."""

from fastapi import FastAPI

from backend.app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Personal AI assistant with a hybrid cloud and local AI architecture.",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Report whether the DEV API process is running."""

    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }
