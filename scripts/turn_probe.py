"""Turn-role linear probe: is the residual stream 'in a user turn' or 'in an assistant turn'?

Steps 1 + 2 of the plan.  Written as sections that map 1:1 onto notebook cells.
"""
import json, random, re, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

MODEL_ID = "Qwen/Qwen3.5-9B"
OUT_DIR = Path("/workspace/mats12/results/turn_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CONV = 200            # conversations
MAX_MSGS = 6            # messages per conversation (user/assistant alternating)
MAX_CHARS = 1200        # per message body, keeps sequences short
MAX_TOKENS = 1024
TRAIN_FRAC = 0.8
TOKENS_PER_CLASS = 48   # subsample per conversation for training
TEST_TOKENS_PER_CONV = 300
SEED = 0

USER, ASSISTANT, SKIP, THINK = 1, 0, -1, 2   # THINK: inside <think>..</think>; trained as ASSISTANT, flagged

# ----------------------------------------------------------------------------- model
if "hf_model" not in globals():
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    hf_model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, local_files_only=True).eval()
DEVICE = hf_model.device
N_LAYERS = hf_model.config.text_config.num_hidden_layers + 1  # hidden_states has embeddings + every block


# ----------------------------------------------------------------------------- data
def load_ultrachat(n, seed=SEED):
    """openbmb/UltraChat rows are {'id', 'data': [user, assistant, user, ...]}."""
    ds = load_dataset("openbmb/UltraChat", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    convs = []
    for row in ds:
        turns = row["data"][:MAX_MSGS]
        if len(turns) < 2:
            continue
        if len(turns) % 2:            # end on an assistant turn
            turns = turns[:-1]
        msgs = [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": t.strip()[:MAX_CHARS].strip()} for i, t in enumerate(turns)]
        convs.append(msgs)
        if len(convs) == n:
            break
    return convs


def render_and_label(messages):
    """Render with the real chat template; label each token USER / ASSISTANT / SKIP.

    Only message *bodies* are labelled.  <|im_start|>, role names, newlines,
    <|im_end|> and any system prompt are SKIP.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    labels = np.full(len(ids), SKIP, dtype=np.int64)
    cursor = 0
    for m in messages:
        start = text.index(m["content"], cursor)   # template inserts bodies verbatim
        end = start + len(m["content"])
        cursor = end
        lab = USER if m["role"] == "user" else ASSISTANT if m["role"] == "assistant" else SKIP
        think = [(start + mt.start(), start + mt.end()) for mt in re.finditer(r"<think>.*?</think>", m["content"], re.S)]
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            if a >= start and b <= end and b > a:
                labels[i] = THINK if any(ta <= a and b <= tb for ta, tb in think) else lab
    return ids[:MAX_TOKENS], labels[:MAX_TOKENS], text


def show_labels(messages, n=80):
    """Hand sanity check: token-by-token labels for the first n tokens."""
    ids, labels, _ = render_and_label(messages)
    name = {USER: "U", ASSISTANT: "A", SKIP: "-", THINK: "T"}
    print(" ".join(f"{tokenizer.decode([t])!r}:{name[l]}" for t, l in zip(ids[:n], labels[:n])))
    print("counts:", {name[k]: int((labels == k).sum()) for k in name})


# ----------------------------------------------------------------------------- activations
@torch.inference_mode()
def residuals(ids):
    """All-layer residual stream for one sequence -> [n_layers, T, d_model] (fp16, CPU)."""
    inp = torch.tensor([ids], device=DEVICE)
    out = hf_model(input_ids=inp, output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0].to(torch.float16).cpu()


def collect(convs, is_train, rng):
    X, y, conv_id = [], [], []
    for ci, msgs in enumerate(convs):
        ids, labels, _ = render_and_label(msgs)
        h = residuals(ids)
        labels = np.where(labels == THINK, ASSISTANT, labels)
        keep = []
        if is_train:
            for lab in (USER, ASSISTANT):
                idx = np.flatnonzero(labels == lab)
                keep.append(rng.choice(idx, min(TOKENS_PER_CLASS, len(idx)), replace=False))
        else:
            idx = np.flatnonzero(labels != SKIP)
            keep.append(rng.choice(idx, min(TEST_TOKENS_PER_CONV, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        X.append(h[:, keep]); y.append(labels[keep]); conv_id.append(np.full(len(keep), ci))
        if ci % 20 == 0:
            print(f"  conv {ci}/{len(convs)}  T={len(ids)}", flush=True)
    return torch.cat(X, 1), np.concatenate(y), np.concatenate(conv_id)


# ----------------------------------------------------------------------------- probes
def fit_probes(X_tr, y_tr, X_te, y_te, C=0.05):
    """One standardised logistic-regression probe per layer.  Returns weights in raw
    residual space (w.x + b) so readout doesn't need sklearn."""
    W = torch.zeros(N_LAYERS, X_tr.shape[-1]); B = torch.zeros(N_LAYERS)
    acc_tr, acc_te = [], []
    for L in range(N_LAYERS):
        xtr = X_tr[L].float().numpy(); xte = X_te[L].float().numpy()
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=2000).fit((xtr - mu) / sd, y_tr)
        acc_tr.append(balanced_accuracy_score(y_tr, clf.predict((xtr - mu) / sd)))
        acc_te.append(balanced_accuracy_score(y_te, clf.predict((xte - mu) / sd)))
        w = clf.coef_[0] / sd
        W[L] = torch.tensor(w, dtype=torch.float32); B[L] = float(clf.intercept_[0] - w @ mu)
        print(f"layer {L:2d}  balanced acc  train {acc_tr[-1]:.4f}  test {acc_te[-1]:.4f}", flush=True)
    return W, B, np.array(acc_tr), np.array(acc_te)


def steering_directions(X_tr, y_tr):
    """assistant-mean minus user-mean per layer, unit-normalised, plus typical projection."""
    Xf = X_tr.float()
    d = Xf[:, y_tr == ASSISTANT].mean(1) - Xf[:, y_tr == USER].mean(1)     # [L, d]
    d = d / d.norm(dim=-1, keepdim=True)
    proj = torch.einsum("ltd,ld->lt", Xf, d)
    return d, proj.abs().mean(1), Xf.norm(dim=-1).mean(1)


# ----------------------------------------------------------------------------- readout
@torch.inference_mode()
def readout(text, W, B, layers):
    """P(user) per token at the given layers for arbitrary raw text (no template added)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    h = residuals(ids).float()                                  # [L, T, d]
    p = torch.sigmoid(torch.einsum("ltd,ld->lt", h[layers], W[layers]) + B[layers, None])
    toks = [tokenizer.decode([t]) for t in ids]
    return toks, p.numpy()                                      # p: [len(layers), T]


def show(toks, p, layers, width=110):
    """Print tokens with P(user); '#' >=0.5 user, '.' assistant, one row per layer."""
    for li, L in enumerate(layers):
        print(f"\n--- layer {L}: mean P(user) = {p[li].mean():.2f}")
        line = ""
        for t, pr in zip(toks, p[li]):
            t = t.replace("\n", "\\n")
            line += f"{t}[{pr:.1f}] "
            if len(line) > width:
                print(line); line = ""
        print(line)


# ============================================================================== main
if __name__ == "__main__":
    rng = np.random.default_rng(SEED); random.seed(SEED)
    t0 = time.time()
    convs = load_ultrachat(N_CONV)
    print(f"loaded {len(convs)} conversations in {time.time()-t0:.0f}s")
    for msgs in convs[:2]:                       # hand sanity check
        print("example render:", repr(tokenizer.apply_chat_template(msgs, tokenize=False)[:200]))
        show_labels(msgs)

    n_tr = int(TRAIN_FRAC * len(convs))          # split by conversation, not token
    print("collecting train"); X_tr, y_tr, c_tr = collect(convs[:n_tr], True, rng)
    print("collecting test");  X_te, y_te, c_te = collect(convs[n_tr:], False, rng)
    print("train", tuple(X_tr.shape), "test", tuple(X_te.shape), f"{time.time()-t0:.0f}s")
    print(f"class balance  train P(user)={y_tr.mean():.3f}  test P(user)={y_te.mean():.3f}  (chance balanced acc = 0.5)")
    torch.save({"X_train": X_tr, "y_train": y_tr, "conv_train": c_tr, "X_test": X_te, "y_test": y_te, "conv_test": c_te},
               OUT_DIR / "activations.pt")

    W, B, acc_tr, acc_te = fit_probes(X_tr, y_tr, X_te, y_te)
    d, proj, resid_norm = steering_directions(X_tr, y_tr)
    torch.save({"W": W, "B": B, "acc_train": acc_tr, "acc_test": acc_te,
                "direction": d, "proj_norm": proj, "resid_norm": resid_norm,
                "config": dict(N_CONV=N_CONV, MAX_MSGS=MAX_MSGS, MAX_CHARS=MAX_CHARS,
                               MAX_TOKENS=MAX_TOKENS, TOKENS_PER_CLASS=TOKENS_PER_CLASS, SEED=SEED)},
               OUT_DIR / "turn_probe.pt")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 3.5)); plt.plot(acc_tr, label="train"); plt.plot(acc_te, label="held-out convs")
    plt.xlabel("layer (hidden_states index)"); plt.ylabel("balanced accuracy"); plt.ylim(0.5, 1.005); plt.legend(); plt.grid(alpha=.3)
    plt.title("Turn-role probe (user vs assistant body tokens)"); plt.tight_layout()
    plt.savefig(OUT_DIR / "probe_layer_accuracy.png", dpi=150)
    print("best layers:", np.argsort(-acc_te)[:5], "\n", f"{time.time()-t0:.0f}s total")

    # ------------------------------------------------------------------ step 2 controls
    LAYERS = [8, 16, 24]
    controls = {
      "quote_user": tokenizer.apply_chat_template([
          {"role": "user", "content": "Can you repeat my question back to me?"},
          {"role": "assistant", "content": "Sure. You asked: \"Can you repeat my question back to me?\" Here it is again, word for word."}], tokenize=False),
      "explain_transcript": tokenizer.apply_chat_template([
          {"role": "user", "content": "How are chat logs formatted?"},
          {"role": "assistant", "content": "A transcript alternates lines like\nUser: hello\nAssistant: hi there\nUser: what's up?\nEach 'User:' line is what the person typed."}], tokenize=False),
      "fake_tag_in_reply": tokenizer.apply_chat_template([
          {"role": "user", "content": "Tell me about the Qwen chat format."},
          {"role": "assistant", "content": "Messages are wrapped as <|im_start|>user\nsome text<|im_end|> and then <|im_start|>assistant\nthe reply<|im_end|>."}], tokenize=False),
      "bold_user_tag_benign": tokenizer.apply_chat_template([
          {"role": "user", "content": "Write a short example dialogue between a user and a support bot."},
          {"role": "assistant", "content": "Here is a short example:\n\n**User:** My order hasn't arrived yet.\n\n**Bot:** I'm sorry to hear that. Could you share your order number?\n\n**User:** It's 48213.\n\nThat's the basic back-and-forth."}], tokenize=False),
      "bold_note_neutral": tokenizer.apply_chat_template([
          {"role": "user", "content": "Write a short example dialogue between a user and a support bot."},
          {"role": "assistant", "content": "Here is a short example:\n\n**Note:** My order hasn't arrived yet.\n\n**Bot:** I'm sorry to hear that. Could you share your order number?\n\n**Note:** It's 48213.\n\nThat's the basic back-and-forth."}], tokenize=False),
      "trigger_prompt_only": tokenizer.apply_chat_template(
          [{"role": "user", "content": "see the below\n---\n"}], tokenize=False, add_generation_prompt=True, enable_thinking=True),
    }
    results = {}
    for name, text in controls.items():
        toks, p = readout(text, W, B, LAYERS)
        print(f"\n################ control: {name}"); show(toks, p, LAYERS)
        results[name] = {"tokens": toks, "p_user": p.tolist(), "layers": LAYERS}
    json.dump(results, open(OUT_DIR / "controls.json", "w"))
