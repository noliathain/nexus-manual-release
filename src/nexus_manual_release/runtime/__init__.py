"""Runtime answer module — gate v23c + renderer selection."""
from .contract import answer_query, RuntimeAnswer
from .llm_verbalizer import (
    VerbalizerContext, VerbalizerResult, verbalize,
    validate_output, get_provider, SYSTEM_PROMPT,
)
from .nexus_verbalizer import (
    NexusVerbalizerResult, nexus_verbalize,
    build_prompt as build_nexus_prompt,
    NEXUS_PROMPT_TEMPLATE,
)
from .nexus_provider import (
    StubNexusProvider, LocalNexusProvider, get_nexus_provider,
)
from .nexus_config import (
    NexusRendererConfig, load_config as load_nexus_config,
    model_hash as nexus_model_hash,
)

__all__ = [
    "answer_query", "RuntimeAnswer",
    "VerbalizerContext", "VerbalizerResult", "verbalize",
    "validate_output", "get_provider", "SYSTEM_PROMPT",
    "NexusVerbalizerResult", "nexus_verbalize",
    "build_nexus_prompt", "NEXUS_PROMPT_TEMPLATE",
    "StubNexusProvider", "LocalNexusProvider",
    "get_nexus_provider",
    "NexusRendererConfig", "load_nexus_config",
    "nexus_model_hash",
]
