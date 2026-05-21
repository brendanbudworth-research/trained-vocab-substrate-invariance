"""Pythia 6.9B-deduped MPS smoke test + tokenization audit.

Gating test before the full Phase 2 cross-model replication. Confirms:
  1. Pythia 6.9B-deduped loads in fp16 on MPS without falling back to CPU
  2. A single forward pass with output_hidden_states succeeds
  3. Throughput is high enough for the full Phase 2 extraction (~16,000
     stimuli total across NEUTRAL + FUNC-PFX × {canonical 500, invented 800})
  4. The tokenizer's subword boundaries for the 10 canonicals + 16 invented
     words are reasonable (no catastrophic splits, no operator that
     becomes an empty/special-token sequence)

While this runs, open Activity Monitor → Window → GPU History.
If GPU usage stays near 0% and CPU pegs to 100% during the forward pass,
MPS is silently falling back. We then switch to fp32 CPU (slow but safe)
or evaluate whether we need to drop Pythia from the replication.

Target throughput on M4: > 80 tok/s for 6.9B fp16. < 40 tok/s = bail.
"""

from __future__ import annotations

import os
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-6.9b-deduped"
PROMPT = "The truth value of p AND q is true when both p and q are"
LAYERS_TO_REPORT = [0, 4, 7, 10, 16, 24, 31]
BATCH_PROMPTS = 8  # Pythia 6.9B is larger; smaller batch than OLMo 1B test

CANONICALS_10 = [
    "and", "or", "implies", "xor", "nand",
    "not", "necessarily", "possibly", "always", "negate",
]

INVENTED_16 = [
    "bliq", "dren", "molex",
    "krev", "sond", "glin", "twiv", "fump",
    "vusp", "perph",
    "kelm", "zorf", "gleph", "drelth", "vrith", "nilph",
]


def proc_rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def tokenization_audit(tok) -> None:
    print()
    print("Tokenization audit (10 canonicals + 16 invented words):")
    print(f"  {'word':<14} {'subwords':<48} {'n':>3}")
    print(f"  {'-'*14} {'-'*48} {'-'*3}")
    for w in CANONICALS_10:
        ids = tok.encode(" " + w, add_special_tokens=False)
        subs = [tok.decode([i]) for i in ids]
        print(f"  {w+' (C)':<14} {str(subs):<48} {len(ids):>3}")
    for w in INVENTED_16:
        ids = tok.encode(" " + w, add_special_tokens=False)
        subs = [tok.decode([i]) for i in ids]
        print(f"  {w+' (I)':<14} {str(subs):<48} {len(ids):>3}")
    print()


def main() -> None:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"RSS before load: {proc_rss_gb():.2f} GB")

    print(f"\nLoading {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    print(f"  load time: {time.time() - t0:.1f}s")
    print(f"  RSS after load: {proc_rss_gb():.2f} GB")

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  config: {n_layers} layers, hidden_size={hidden_dim}, "
          f"vocab_size={model.config.vocab_size}")
    print(f"  architectures: {model.config.architectures}")
    layers = [l for l in LAYERS_TO_REPORT if l < n_layers + 1]

    tokenization_audit(tok)

    print("Single-prompt forward pass (no hidden states; warmup):")
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
    print(f"  num hidden_states: {len(out.hidden_states)} "
          f"(should be num_layers + 1 = {n_layers + 1})")
    for l in layers:
        if l < len(out.hidden_states):
            hs = out.hidden_states[l]
            print(f"  layer {l:2d}: shape={tuple(hs.shape)} "
                  f"dtype={hs.dtype} device={hs.device}")

    print(f"\nBatched throughput test ({BATCH_PROMPTS} prompts; hidden states on):")
    enc = tok([PROMPT] * BATCH_PROMPTS, return_tensors="pt", padding=True).to(device)
    t0 = time.time()
    with torch.no_grad():
        model(**enc, output_hidden_states=True)
    if device == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    n_tok = int(enc.attention_mask.sum().item())
    tok_per_s = n_tok / dt
    print(f"  batch forward: {dt:.2f}s for {n_tok} attended tokens "
          f"({tok_per_s:.0f} tok/s)")
    print(f"  RSS after run: {proc_rss_gb():.2f} GB")

    n_canon_stim_estimated = 2 * 10 * 50  # 2 conds × 10 canonicals × 50 per class
    n_inv_stim_estimated = 2 * 16 * 50    # 2 conds × 16 invented × 50 per class
    n_total = n_canon_stim_estimated + n_inv_stim_estimated
    avg_tok_per_stim = 30  # rough estimate
    est_total_tok = n_total * avg_tok_per_stim
    est_extract_sec = est_total_tok / max(tok_per_s, 1)

    print()
    print(f"Extraction time estimate (single-stim forward, not batched):")
    print(f"  total stimuli to extract: ~{n_total} "
          f"({n_canon_stim_estimated} canonical + {n_inv_stim_estimated} invented)")
    print(f"  at observed {tok_per_s:.0f} tok/s, with avg ~{avg_tok_per_stim} tok/stim: "
          f"~{est_extract_sec / 60:.1f} min")
    print(f"  (single-stim extraction is slower than this batched estimate;")
    print(f"   the actual extraction at batch=1 is typically 2-3x slower per token)")
    print(f"  realistic estimate: ~{est_extract_sec * 2.5 / 60:.0f}-"
          f"{est_extract_sec * 3 / 60:.0f} minutes")

    print()
    print("Decision criteria:")
    print("  GREEN: throughput > 80 tok/s AND GPU History showed sustained activity")
    print("         → proceed with full replication on MPS fp16")
    print("  YELLOW: throughput 40-80 tok/s — viable but slow; consider running overnight")
    print("  RED: throughput < 40 tok/s OR GPU stayed near 0% — drop to fp32 CPU")
    print("       (which will be ~5-10× slower, so plan a multi-hour run)")


if __name__ == "__main__":
    main()
