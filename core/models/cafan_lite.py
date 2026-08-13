"""Import bridge for optional CAFAN models kept in the HY experiment folder."""

import os
import sys

from core.utils.registry import registry


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

try:
    from HY.cafan import CAFANLite  # noqa: F401
except ModuleNotFoundError as exc:
    @registry.register_model("cafan_lite")
    class CAFANLite:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "Model 'cafan_lite' requires the optional module 'HY.cafan', "
                "which is not available in this environment."
            ) from exc
else:
    if registry.get_model_class("cafan_lite") is None:
        registry.register_model("cafan_lite")(CAFANLite)
