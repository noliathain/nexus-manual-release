"""Nexus model integration.

Imports inside this package are lazy: pulling ``modeling`` does *not* import
PyTorch. The CLI's ``mgr demo`` path works without the ``train`` extra
installed. Build the actual ``nn.Module`` via
``modeling.nexus_config.create_nexus_manual_rag_8m_512_config`` +
``modeling.nexus`` only when training/inference is required.
"""
