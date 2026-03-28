from .easy import get_easy_state
from .medium import get_medium_state
from .hard import get_hard_state

TASK_REGISTRY = {
    "easy_api_crash": get_easy_state,
    "medium_cache_latency": get_medium_state,
    "hard_cascading_incident": get_hard_state,
}
