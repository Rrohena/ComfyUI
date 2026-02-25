"""
ip_adapter_lite — lightweight IP-Adapter conditioning for SD1.5 on low-VRAM GPUs.

Nodes:
  • IPAdapterLiteModelLoader  — loads IPA weights from models/ipadapter/
  • IPAdapterLiteApply        — patches a MODEL with image reference conditioning
"""

from .ip_adapter_lite import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
