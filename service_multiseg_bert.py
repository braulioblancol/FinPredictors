from flask import Flask, request, jsonify
import argparse
import time
from bert_multiseg_dynamic import PageTypeInference  # import your class

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./BEST_MODEL")
args = parser.parse_args()


# Initialize Flask app
app = Flask(__name__)

# Load the model once when the server starts
inference = PageTypeInference(model_dir=args.model_dir)

@app.route("/predict", methods=["POST"])
def predict():
    time_start = time.time()
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    try:
        result = inference.predict(text)
        time_elapsed = (time.time() - time_start) 
        result['response_time'] = time_elapsed
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch_endpoint():
    time_start = time.time()
    data = request.json
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400
    results = inference.predict_batch(data["texts"])
    time_elapsed = (time.time() - time_start) 

    for result in results:
        result['response_time'] = time_elapsed / len(results)
    return jsonify(results)


# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=False)
