"""Every public entry point imports cleanly under HF_HUB_OFFLINE=1."""
from __future__ import annotations


def test_package_imports():
    import nexus_manual_release
    assert hasattr(nexus_manual_release, "__version__")

def test_runtime_imports():
    from nexus_manual_release.runtime import (
        answer_query, RuntimeAnswer, VerbalizerContext,
        validate_output, get_nexus_provider,
        nexus_model_hash,
    )
    assert callable(answer_query)
    assert callable(validate_output)
    assert callable(get_nexus_provider)

def test_cli_imports():
    from nexus_manual_release.cli import app, main
    assert app is not None
    assert callable(main)

def test_modeling_lazy_import():
    # Importing the package must NOT trigger torch import.
    import nexus_manual_release.modeling
    assert nexus_manual_release.modeling

def test_encoder_resolves_to_bundled_path():
    from nexus_manual_release.runtime.semantic_retrieval \
        import DEFAULT_ENCODER, _BUNDLED_ENCODER_DIR
    # On a fresh clone the bundled dir must win — env vars are
    # cleared at this point and the bundled dir has the
    # safetensors file in it.
    assert str(_BUNDLED_ENCODER_DIR) in DEFAULT_ENCODER
    assert "minishlab" not in DEFAULT_ENCODER
