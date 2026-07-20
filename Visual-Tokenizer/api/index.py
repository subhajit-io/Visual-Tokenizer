from flask import Flask, request, jsonify
from flask_cors import CORS
import regex as re

app = Flask(__name__)
CORS(app)

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

TOKENIZER_PATTERN = re.compile("|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    ))

def tokenizer(text):
    tokens = re.findall(TOKENIZER_PATTERN, text)
    result = []
    for token in tokens:
        if token == '\n':
            result.append("'\\n")  
        else:
            result.append(f"'{token}'")
    return result


@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    try:
        data = request.json
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        tokens = tokenizer(text)
        return jsonify({"success": True, "text": text, "tokens": tokens, "count": len(tokens)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/api/bpetokenize", methods=["POST"])
def api_bpetokenize():
    try:
        data = request.json
        test_text = data.get("text", "") 
        training_text = data.get("training_text", "").strip()
        
        train_merges = int(data.get("train_merges", 0))
        test_merges = int(data.get("test_merges", 0))
        
        if not test_text:
            return jsonify({"error": "No text provided"}), 400
            
        if not training_text:
            training_text = test_text
        
        train_tokens = tokenizer(training_text)
        test_tokens = tokenizer(test_text)
        
        train_bytes = []
        for t in train_tokens:
            clean_token = '\n' if t == "'\\n" else t.strip("'")
            train_bytes.append(list(clean_token.encode("utf-8")))
            
        merges = {}
        merge_history = [] 
        vocab = {i: bytes([i]) for i in range(256)}
        
        current_id = 256

        special_rules = [
            ((32, 32), "Double Space"),
            ((10, 10), "Double Newline")
        ]
        
        for pair, label in special_rules:
            merges[pair] = current_id
            vocab[current_id] = vocab[pair[0]] + vocab[pair[1]]
            merge_history.append(f"{list(pair)} \u2794 {current_id} [{label}]")

            for j in range(len(train_bytes)):
                train_bytes[j] = merge_ids(train_bytes[j], pair, current_id)
                
            current_id += 1

        for i in range(train_merges):
            stats = {}
            for chunk in train_bytes:
                for pair in zip(chunk, chunk[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
                    
            if not stats:
                break 
                
            best_pair = max(stats, key=stats.get)
            idx = current_id
            merges[best_pair] = idx 
            
            vocab[idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merge_history.append(f"{list(best_pair)} \u2794 {idx}")
            
            for j in range(len(train_bytes)):
                train_bytes[j] = merge_ids(train_bytes[j], best_pair, idx)
                
            current_id += 1
                
        allowed_max_id = 255 + len(special_rules) + test_merges
        active_merges = {k: v for k, v in merges.items() if v <= allowed_max_id}
        
        test_bytes = []
        for t in test_tokens:
            clean_token = '\n' if t == "'\\n" else t.strip("'")
            chunk = list(clean_token.encode("utf-8"))
            
            while len(chunk) >= 2:
                stats = {}
                for pair in zip(chunk, chunk[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
                if not stats:
                    break
                    
                pair_to_merge = min(stats.keys(), key=lambda p: active_merges.get(p, float('inf')))
                if pair_to_merge not in active_merges:
                    break
                    
                idx = active_merges[pair_to_merge]
                chunk = merge_ids(chunk, pair_to_merge, idx)
                
            detailed_chunk = []
            for tid in chunk:
                text_representation = vocab[tid].decode("utf-8", errors="replace")
                detailed_chunk.append({"id": tid, "text": text_representation})
                
            test_bytes.append(detailed_chunk)
        
        return jsonify({
            "success": True,
            "text": test_text,
            "bpe_bytes": test_bytes,
            "learned_merges": merge_history 
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Backend is running"}), 200

#for local testing
if __name__ == "__main__":
    print(">>> Python backend running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
