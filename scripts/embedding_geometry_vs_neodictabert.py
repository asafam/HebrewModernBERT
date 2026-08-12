"""Embedding-geometry diagnostic: why HMB transfers to retrieval worse than NeoDictaBERT.

CONTEXT
-------
scripts/mlm_compare_vs_neodictabert.py established that HMB retrain-final and NeoDictaBERT are
at parity on masked language modelling (48.6% vs 49.9% whole-word recovery) yet score 0.185 vs
0.332 on BeIR retrieval through the identical SFT pipeline. Equivalent language knowledge,
wildly different retrieval => the loss happens when representations are turned into a single
retrievable vector. This script measures that step directly, with no training involved.

WHAT IT MEASURES, per model x pooling x layer
---------------------------------------------
  ndcg10      zero-shot NDCG@10 using raw embeddings. How much retrieval signal is ALREADY in
              this layer before any SFT. The bottom line.
  aniso       mean pairwise cosine between random document embeddings. The "background
              similarity floor". ~0 is a healthy spread; ->1 means the space has collapsed
              into a narrow cone and nothing can be discriminated.
  align       mean cosine between a query and its known-relevant document (from qrels).
  SEP         align - aniso. The actual usable signal: how much closer a true positive sits
              than an arbitrary document. A model can have high `align` and still retrieve
              nothing if `aniso` is just as high. This is the number that matters.

WHY PER-LAYER
-------------
MLM training specializes the final layers for token reconstruction, which can destroy
sentence-level geometry that earlier layers still have. If an intermediate HMB layer shows
markedly better SEP/ndcg10 than the last, that is a near-free fix (pool from that layer during
SFT) versus weeks of retraining.

WHY BOTH POOLINGS
-----------------
Our locked SFT protocol uses CLS; NeoDictaBERT's eval used mean. If HMB's CLS geometry is
collapsed while its mean geometry is healthy, we have been handicapping ourselves at the
interface, not in the model.
"""
import argparse, glob, json, os, random, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BEIR = ("/home/nlp/achimoa/workspace/hebrew_text_retrieval/outputs/translation/runs/"
        "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")
HF = "/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf"

MODELS = {
    "NeoDictaBERT":           "dicta-il/NeoDictaBERT",
    "HMB retrain FINAL (p2)": f"{HF}/HebrewModernBERT-base-retrain-final",
    "HMB phase-0 FINAL":      f"{HF}/HebrewModernBERT-base-phase0-arm1-m20-final",
    "HMB random-init":        f"{HF}/HebrewModernBERT-base-random",
}
# Small corpora, and the two where we trail NeoDictaBERT worst in absolute terms.
DATASETS = ["scifact", "nfcorpus"]


def load_ds(ds):
    d = f"{BEIR}/BeIR_{ds}/beir"
    corpus, queries = {}, {}
    with open(f"{d}/corpus.jsonl") as f:
        for line in f:
            o = json.loads(line)
            corpus[o["_id"]] = ((o.get("title") or "") + " " + (o.get("text") or "")).strip()
    with open(f"{d}/queries.jsonl") as f:
        for line in f:
            o = json.loads(line)
            queries[o["_id"]] = (o.get("text") or "").strip()
    qrels = {}
    with open(f"{d}/qrels/test.jsonl") as f:
        for line in f:
            o = json.loads(line)
            if o.get("score", 0) > 0 and o["corpus-id"] in corpus:
                qrels.setdefault(o["query-id"], set()).add(o["corpus-id"])
    # only queries that have at least one judged positive, matching the main eval
    qids = [q for q in qrels if q in queries and queries[q]]
    return corpus, queries, qrels, sorted(qids)


@torch.no_grad()
def encode_all_layers(model, tok, texts, device, bs, max_len):
    """-> dict {pooling: FloatTensor [n_layers, n_texts, dim]} (fp16, on CPU)."""
    out = {"mean": [], "cls": []}
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple[L+1] of [B,T,H]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        denom = mask.sum(1).clamp(min=1e-9)
        out["mean"].append(torch.stack([ (h * mask).sum(1) / denom for h in hs ]).half().cpu())
        out["cls"].append(torch.stack([ h[:, 0] for h in hs ]).half().cpu())
    return {k: torch.cat(v, dim=1) for k, v in out.items()}


def ndcg_at_10(qemb, demb, qids, doc_ids, qrels):
    """Cosine retrieval -> NDCG@10. IDCG truncated at 10 (matches the fixed main eval)."""
    q = F.normalize(qemb.float(), dim=-1)
    d = F.normalize(demb.float(), dim=-1)
    idx = {c: i for i, c in enumerate(doc_ids)}
    sims = q @ d.T
    k = min(10, sims.shape[1])
    top = sims.topk(k, dim=-1).indices
    total = 0.0
    for row, qid in enumerate(qids):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        gains = [1.0 if doc_ids[j] in rel else 0.0 for j in top[row].tolist()]
        dcg = sum(g / torch.log2(torch.tensor(r + 2.0)).item() for r, g in enumerate(gains))
        n_ideal = min(len(rel), k)
        idcg = sum(1.0 / torch.log2(torch.tensor(r + 2.0)).item() for r in range(n_ideal))
        total += dcg / idcg if idcg > 0 else 0.0
    return total / max(len(qids), 1)


def geometry(qemb, demb, qids, doc_ids, qrels, n_sample=1500, seed=17):
    """aniso = mean pairwise cos among random docs; align = mean cos(query, its positive)."""
    d = F.normalize(demb.float(), dim=-1)
    rng = random.Random(seed)
    sel = rng.sample(range(d.shape[0]), min(n_sample, d.shape[0]))
    s = d[sel]
    sim = s @ s.T
    n = sim.shape[0]
    aniso = ((sim.sum() - sim.diag().sum()) / (n * (n - 1))).item()

    q = F.normalize(qemb.float(), dim=-1)
    idx = {c: i for i, c in enumerate(doc_ids)}
    vals = []
    for row, qid in enumerate(qids):
        for c in qrels.get(qid, ()):
            if c in idx:
                vals.append(torch.dot(q[row], d[idx[c]]).item())
    align = sum(vals) / len(vals) if vals else float("nan")
    return aniso, align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_docs", type=int, default=6000)
    ap.add_argument("--out", default="outputs/eval/embedding_geometry.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = {}
    for ds in DATASETS:
        corpus, queries, qrels, qids = load_ds(ds)
        doc_ids = list(corpus)[:args.max_docs]
        keep = set(doc_ids)
        # keep every judged positive in the pool even if truncation would drop it
        for q in qids:
            for c in qrels[q]:
                if c not in keep and c in corpus:
                    doc_ids.append(c); keep.add(c)
        data[ds] = (corpus, queries, qrels, qids, doc_ids)
        print(f"  {ds:9} {len(doc_ids)} docs, {len(qids)} judged queries")

    results = {}
    for label, path in MODELS.items():
        if not path.startswith("dicta-il") and not os.path.isdir(path):
            print(f"  SKIP {label}"); continue
        print(f"\n=== {label} ===", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
            model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()
        except Exception as exc:
            print(f"  FAILED to load: {type(exc).__name__}: {exc}"); continue

        results[label] = {}
        for ds in DATASETS:
            corpus, queries, qrels, qids, doc_ids = data[ds]
            demb = encode_all_layers(model, tok, [corpus[c] for c in doc_ids],
                                     device, args.batch_size, args.max_length)
            qemb = encode_all_layers(model, tok, [queries[q] for q in qids],
                                     device, args.batch_size, args.max_length)
            nlayers = demb["mean"].shape[0]
            results[label][ds] = {}
            for pool in ("mean", "cls"):
                per_layer = []
                for L in range(nlayers):
                    nd = ndcg_at_10(qemb[pool][L], demb[pool][L], qids, doc_ids, qrels)
                    an, al = geometry(qemb[pool][L], demb[pool][L], qids, doc_ids, qrels)
                    per_layer.append({"layer": L, "ndcg10": nd, "aniso": an,
                                      "align": al, "sep": al - an})
                results[label][ds][pool] = per_layer
                best = max(per_layer, key=lambda r: r["ndcg10"])
                last = per_layer[-1]
                print(f"  {ds:9} {pool:4}  last(L{last['layer']}): ndcg={last['ndcg10']:.4f} "
                      f"aniso={last['aniso']:+.3f} align={last['align']:+.3f} SEP={last['sep']:+.3f}"
                      f"   |  best L{best['layer']}: ndcg={best['ndcg10']:.4f} SEP={best['sep']:+.3f}",
                      flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    for ds in DATASETS:
        for pool in ("mean", "cls"):
            print(f"\n=== {ds} / {pool} pooling — LAST layer (what SFT actually consumes) ===")
            print(f"{'model':26} {'ndcg@10':>9} {'aniso':>8} {'align':>8} {'SEP':>8}   best-layer")
            print("-" * 78)
            for label in MODELS:
                if label not in results or ds not in results[label]:
                    continue
                pl = results[label][ds][pool]
                last, best = pl[-1], max(pl, key=lambda r: r["ndcg10"])
                gain = f"L{best['layer']} ({best['ndcg10']:.4f}"
                gain += f", +{best['ndcg10']-last['ndcg10']:.4f})" if best["layer"] != last["layer"] else ", =last)"
                print(f"{label:26} {last['ndcg10']:9.4f} {last['aniso']:8.3f} "
                      f"{last['align']:8.3f} {last['sep']:8.3f}   {gain}")

    print("""
HOW TO READ
  SEP is the diagnostic. If NeoDictaBERT has a much larger SEP than HMB at the last layer,
  our embedding space is collapsed relative to theirs and that -- not language knowledge --
  is the retrieval gap. If some intermediate HMB layer has a markedly higher SEP/ndcg10 than
  its last layer, pooling from that layer during SFT is a cheap win. If HMB's mean pooling is
  far healthier than its cls, the locked cls SFT protocol is costing us for free.""")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
