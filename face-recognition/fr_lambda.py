# fr_lambda.py
import os
import json
import boto3
import base64
import tempfile
import torch
import numpy as np
from PIL import Image

# If you load MTCNN elsewhere, leave it out here; recognition uses the ResNet.
# from facenet_pytorch import MTCNN  # Not required for recognition

# ---------- Configuration via Environment Variables ----------
REGION = os.getenv("AWS_REGION", "us-east-1")
SQS_RESPONSE_QUEUE_URL = os.getenv(
    "SQS_RESPONSE_QUEUE_URL",
    "https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/your-resp-queue"
)

# Model artifacts baked into the Lambda package/container image
MODEL_PATH = os.getenv("MODEL_PATH", "resnetV1.pt")
MODEL_WT_PATH = os.getenv("MODEL_WT_PATH", "resnetV1_video_weights.pt")
# -------------------------------------------------------------

sqs = boto3.client("sqs", region_name=REGION)

class FaceRecognition:
    def __init__(self):
        # Lazy-load model on first invocation to reduce cold start time on small images
        self._resnet = None
        self._embeddings = None
        self._names = None

    def _load_models(self):
        if self._resnet is None:
            # TorchScript model recommended for Lambda
            self._resnet = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
        if self._embeddings is None or self._names is None:
            saved = torch.load(MODEL_WT_PATH, map_location="cpu")
            self._embeddings, self._names = saved[0], saved[1]

    def predict_name(self, face_img_path: str) -> str:
        self._load_models()

        face_pil = Image.open(face_img_path).convert("RGB")
        face_np = np.array(face_pil, dtype=np.float32) / 255.0
        face_np = np.transpose(face_np, (2, 0, 1))  # HWC -> CHW
        face_tensor = torch.tensor(face_np, dtype=torch.float32).unsqueeze(0)

        with torch.inference_mode():
            emb = self._resnet(face_tensor).detach()

        # L2 distances to stored embeddings
        dists = [torch.dist(emb, e).item() for e in self._embeddings]
        idx = int(np.argmin(dists))
        return self._names[idx]

recognizer = FaceRecognition()

def lambda_handler(event, context):
    """
    Triggered by SQS (request queue). For each record:
      - decodes face image,
      - predicts identity,
      - sends {request_id, result} to response queue.
    """
    # SQS batch events
    for record in event.get("Records", []):
        try:
            payload = json.loads(record["body"])
            request_id = payload["request_id"]
            face_b64 = payload["face"]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as tmpf:
                tmpf.write(base64.b64decode(face_b64))
                face_path = tmpf.name

            label = recognizer.predict_name(face_path)

            result_msg = {
                "request_id": request_id,
                "result": label
            }
            sqs.send_message(QueueUrl=SQS_RESPONSE_QUEUE_URL, MessageBody=json.dumps(result_msg))
        except Exception as e:
            # Log and continue to next record. Let SQS redrive handle retries if configured.
            print(f"[ERROR] Failed to process record: {e}")
        finally:
            try:
                if "face_path" in locals() and os.path.exists(face_path):
                    os.remove(face_path)
            except OSError:
                pass

    return {"statusCode": 200, "body": json.dumps({"message": "Face recognition completed."})}
