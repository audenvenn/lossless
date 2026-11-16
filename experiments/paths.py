"""Model and engine locations, from environment variables (no machine-specific paths in the code).

  LOSSLESS_QWEN_BF16   Hugging Face checkpoint dir (or repo id) of Qwen3.5-4B with its MTP layer   [Qwen/Qwen3.5-4B]
  LOSSLESS_QWEN_MLX4   MLX 4-bit build used as the fixed reference scorer                      [mlx-community/Qwen3.5-4B-4bit]
  LOSSLESS_TOKENIZER   tokenizer for identity checks                                         [= LOSSLESS_QWEN_BF16]
  RACELINE_SRC         raceline source checkout (for the sampler L1 test)                    [unset]
  RACELINE_LIB         path of libraceline.dylib (raceline's C top-k)                        [unset]

A local, untracked `paths.local.env` can export these before running the scripts.
"""
import os
import sys

for _k in ("LOSSLESS_QWEN_BF16", "LOSSLESS_QWEN_MLX4"):
    if _k not in os.environ:
        print(f"[paths] {_k} not set: falling back to a Hugging Face repo id (set it, or source paths.local.env; "
              f"the run_*.sh scripts set HF_HUB_OFFLINE=1)", file=sys.stderr)

QWEN_BF16 = os.environ.get("LOSSLESS_QWEN_BF16", "Qwen/Qwen3.5-4B")
QWEN_MLX4 = os.environ.get("LOSSLESS_QWEN_MLX4", "mlx-community/Qwen3.5-4B-4bit")
TOKENIZER = os.environ.get("LOSSLESS_TOKENIZER", QWEN_BF16)
RACELINE_SRC = os.environ.get("RACELINE_SRC")
RACELINE_LIB = os.environ.get("RACELINE_LIB")
