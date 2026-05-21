"""MPS smoke test on OLMo 2 1B.

Loads OLMo 2 1B in fp16, runs a forward pass with hidden-state output,
and reports throughput. This is the gating test for whether Phase 0
can run locally on the M4 or must be done cluster-first.

While this runs, open Activity Monitor → Window → GPU History.
If GPU usage stays near 0% and CPU pegs to 100% during the forward
pass, MPS is silently falling back. Bail to cluster-first.
"""

from __future__ import annotations

import os
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B"
PROMPT = "The truth value of p AND q is true when both p and q are"
LAYERS_TO_REPORT = [0, 4, 8, 12, 15]
BATCH_PROMPTS = 32


def proc_rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def main() -> None:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"RSS before load: {proc_rss_gb():.2f} GB")

    print(f"\nLoading {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(device).eval()
    print(f"  load time: {time.time() - t0:.1f}s")
    print(f"  RSS after load: {proc_rss_gb():.2f} GB")

    n_layers = model.config.num_hidden_layers
    print(f"  num layers: {n_layers}")
    layers = [l for l in LAYERS_TO_REPORT if l < n_layers]

    print("\nSingle-prompt forward pass (no hidden states):")
    inputs = tok(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs)
    if device == "mps":
        torch.mps.synchronize()
    print(f"  forward time: {(time.time() - t0) * 1000:.0f} ms")
    print(f"  sequence length: {inputs.input_ids.shape[1]}")

    print("\nSingle-prompt forward pass with hidden states:")
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    if device == "mps":
        torch.mps.synchronize()
    print(f"  forward time: {(time.time() - t0) * 1000:.0f} ms")
    for l in layers:
        hs = out.hidden_states[l]
        print(f"  layer {l:2d}: shape={tuple(hs.shape)} dtype={hs.dtype} device={hs.device}")

    print(f"\nBatched throughput test ({BATCH_PROMPTS} prompts):")
    enc = tok([PROMPT] * BATCH_PROMPTS, return_tensors="pt", padding=True).to(device)
    t0 = time.time()
    with torch.no_grad():
        model(**enc, output_hidden_states=True)
    if device == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    n_tok = int(enc.attention_mask.sum().item())
    print(f"  batch forward: {dt:.2f}s for {n_tok} attended tokens ({n_tok / dt:.0f} tok/s)")
    print(f"  RSS after run: {proc_rss_gb():.2f} GB")

    print("\nDecision criteria:")
    print("  GREEN: throughput > 200 tok/s AND GPU History showed sustained activity")
    print("  YELLOW: throughput 50-200 tok/s — local viable but slow; profile before committing")
    print("  RED: throughput < 50 tok/s OR GPU stayed near 0% — switch to cluster-first")


if __name__ == "__main__":
    main()
