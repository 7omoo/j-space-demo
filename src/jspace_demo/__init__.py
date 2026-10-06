"""J-Space Demo: what an LLM says, and what it has in mind, read with the Jacobian lens (J-lens).

Importing jspace_demo turns on PyTorch's CPU fallback for operators the MPS backend lacks. PyTorch reads
PYTORCH_ENABLE_MPS_FALLBACK once, when torch is first imported, so jspace_demo has to be imported before torch,
and before jlens or transformers, which import torch themselves.
"""

import os
import sys

# False when torch was already imported without the variable set: the fallback is then off for this process.
FALLBACK_READY = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0" or "torch" not in sys.modules
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
