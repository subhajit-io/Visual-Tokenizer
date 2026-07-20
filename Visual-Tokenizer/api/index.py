from flask import Flask, request, jsonify
from flask_cors import CORS
import regex as re

app = Flask(__name__)
CORS(app)

# --- CACHE STORAGE ---
# This stores the results so we don't have to re-train on every keystroke
cache = {
    "last_train_text": None,
    "last_merges": {},
    "last_vocab": {},
    "last_history": []
}

def merge_ids(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]: 
            newids.append(idx)
            i += 2 
        else:
            newids.append(ids[i])
            i += 1
    return newids

# [Keep your existing TOKENIZER_PATTERN and tokenizer function here...]
TOKENIZER_PATTERN = re.compile("|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
]))

def tokenizer(text):
    tokens = re.findall(TOKENIZER_PATTERN, text)
    result = []
    for token in tokens:
        if token == '\n':
            result.append("'\\n")  
        else:
            result.append(f"'{token}'")
    return result

@app.route("/api/bpetokenize", methods=["POST"])
def api_bpetokenize():
    try:
        data = request.json
        test_text = data.get("text", "") 
        training_text = data.get("training_text", "").strip()
        train_merges = int(data.get("train_merges", 0))
        test_merges = int(data.get("test_merges", 0))

        # --- CACHING LOGIC ---
        if training_text != cache["last_train_text"]:
            # Only run the heavy training if the training text changed
            train_tokens = tokenizer(training_text if training_text else test_text)
            train_bytes = [list(('\n' if t == "'\\n" else t.strip("'")).encode("utf-8")) for t in train_tokens]
            
            merges = {}
            merge_history = []
            vocab = {i: bytes([i]) for i in range(256)}
            current_id = 256
            
            # Special rules
            for pair, label in [((32, 32), "Double Space"), ((10, 10), "Double Newline")]:
                merges[pair] = current_id
                vocab[current_id] = vocab[pair[0]] + vocab[pair[1]]
                merge_history.append(f"{list(pair)} \u2794 {current_id} [{label}]")
                for j in range(len(train_bytes)): train_bytes[j] = merge_ids(train_bytes[j], pair, current_id)
                current_id += 1

            # Training loop
            for _ in range(train_merges):
                stats = {}
                for chunk in train_bytes:
                    for pair in zip(chunk, chunk[1:]): stats[pair] = stats.get(pair, 0) + 1
                if not stats: break
                best_pair = max(stats, key=stats.get)
                merges[best_pair] = current_id
                vocab[current_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
                merge_history.append(f"{list(best_pair)} \u2794 {current_id}")
                for j in range(len(train_bytes)): train_bytes[j] = merge_ids(train_bytes[j], best_pair, current_id)
                current_id += 1
            
            # Update cache
            cache.update({"last_train_text": training_text, "last_merges": merges, "last_vocab": vocab, "last_history": merge_history})

        # --- TESTING PHASE (Fast) ---
        active_merges = {k: v for k, v in cache["last_merges"].items() if v <= 255 + 2 + test_merges}
        test_tokens = tokenizer(test_text)
        test_bytes = []
        for t in test_tokens:
            chunk = list(('\n' if t == "'\\n" else t.strip("'")).encode("utf-8"))
            while len(chunk) >= 2:
                stats = {p: 0 for p in zip(chunk, chunk[1:])}
                if not stats: break
                pair_to_merge = min(stats.keys(), key=lambda p: active_merges.get(p, float('inf')))
                if pair_to_merge not in active_merges: break
                chunk = merge_ids(chunk, pair_to_merge, active_merges[pair_to_merge])
            test_bytes.append([{"id": tid, "text": cache["last_vocab"][tid].decode("utf-8", errors="replace")} for tid in chunk])

        return jsonify({"success": True, "bpe_bytes": test_bytes, "learned_merges": cache["last_history"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
