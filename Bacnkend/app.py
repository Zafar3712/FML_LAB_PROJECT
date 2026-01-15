from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import sqlite3
from io import BytesIO
from PIL import Image
from utils.face_embedder import get_embedding

app = Flask(__name__)
CORS(app)

model = load_model("model/embedding_model.h5", safe_mode=False)
THRESHOLD = 0.7 # If distance is > 0.7, it's not a match

def connect_db():
    return sqlite3.connect("faces.db", check_same_thread=False)

@app.route("/register", methods=["POST"])
def register_face():
    data = request.json
    # --- ADD THIS CHECK ---
    if not data or 'image' not in data or data['image'] is None:
        return jsonify({"message": "No image data sent."}), 400
    # --- END OF CHECK ---

    name = data["name"]
    img_data = base64.b64decode(data["image"].split(",")[1])
    image = np.array(Image.open(BytesIO(img_data)).convert("RGB"))
    embedding = get_embedding(model, image)

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS faces (name TEXT, embedding BLOB)")
    cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding.tobytes()))
    conn.commit()
    conn.close()

    return jsonify({"message": f"Face registered for {name}!"})

@app.route("/recognize", methods=["POST"])
def recognize_face():
    data = request.json
    # --- ADD THIS CHECK ---
    if not data or 'image' not in data or data['image'] is None:
        return jsonify({"identity": "Unknown", "distance": -1, "message": "No image data sent."}), 400
    # --- END OF CHECK ---


    img_data = base64.b64decode(data["image"].split(",")[1])
    image = np.array(Image.open(BytesIO(img_data)).convert("RGB"))
    query_embedding = get_embedding(model, image)

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM faces")
    rows = cursor.fetchall()
    conn.close()

    min_dist = float("inf")
    identity = "Unknown"
    for name, emb_blob in rows:
        stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
        dist = np.linalg.norm(stored_emb - query_embedding)
        if dist < min_dist:
            min_dist = dist
            identity = name

    # --- REPLACE THE OLD RETURN STATEMENT WITH THIS ---
    if min_dist < THRESHOLD:
        # The match is good enough
        return jsonify({
            "identity": identity,
            "distance": float(min_dist)
        })
    else:
        # The match is too poor, so we don't recognize them
        return jsonify({
            "identity": "Not Recognized",
            "distance": float(min_dist)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
