import uvicorn
from fastapi import APIRouter, FastAPI

from bsllmner2.config import MODULE_ROOT, get_config, set_logging_level

# === API Router ===

router = APIRouter()


@router.get(
    "/service-info",
)
async def service_info():
    return {"service": "bsllmner2", "version": "1.0.0"}


# === main application setup ===

def create_app() -> FastAPI:
    app_config = get_config()
    set_logging_level(app_config.debug)  # Reconfigure logging level

    app = FastAPI(
        root_path=app_config.api_url_prefix,
        debug=app_config.debug,
    )

    app.include_router(router)

    return app


def run_api() -> None:
    app_config = get_config()
    set_logging_level(app_config.debug)
    uvicorn.run(
        "bsllmner2.api:create_app",
        host=app_config.api_host,
        port=app_config.api_port,
        reload=app_config.debug,
        reload_dirs=[str(MODULE_ROOT)] if app_config.debug else None,
        factory=True,
    )


if __name__ == "__main__":
    run_api()
