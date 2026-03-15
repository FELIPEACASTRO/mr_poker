from apps.api.routers.benchmark import router as benchmark_router
from apps.api.routers.governance import router as governance_router
from apps.api.routers.hands import router as hands_router
from apps.api.routers.ml import router as ml_router
from apps.api.routers.sessions import router as sessions_router
from apps.api.routers.spots import router as spots_router
from apps.api.routers.system import router as system_router

__all__ = [
    "benchmark_router",
    "governance_router",
    "hands_router",
    "ml_router",
    "sessions_router",
    "spots_router",
    "system_router",
]
