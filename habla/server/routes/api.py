"""REST API routes — re-export aggregator.

Domain logic lives in the api_*.py modules. This file re-exports all routers
and setter functions so existing imports from ``server.routes.api`` keep working.
"""

# Routers
from server.routes.api_vocab import vocab_router  # noqa: F401
from server.routes.api_system import system_router  # noqa: F401
from server.routes.api_sessions import session_router  # noqa: F401
from server.routes.api_idioms import idiom_router, _generate_pattern  # noqa: F401
from server.routes.api_llm import llm_router, lmstudio_router  # noqa: F401
from server.routes.api_playback import playback_router  # noqa: F401

# State setters (called from main.py lifespan)
from server.routes._state import set_pipeline, set_lmstudio_manager, set_playback_service  # noqa: F401
