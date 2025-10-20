# fd_component.py
import json
import base64
import boto3
import logging
import os
import sys
import threading
from io import BytesIO
from PIL import Image
import numpy as np
from awscrt import mqtt
from awsiot import mqtt_connection_builder
from facenet_pytorch import MTCNN

# ---------- Configuration via Environment Variables ----------
REGION = os.getenv("AWS_REGION", "us-east-1")

# IoT / MQTT
THING_NAME = os.getenv("IOT_THING_NAME", "your-iot-thing-name")
TOPIC = os.getenv("IOT_TOPIC", f"clients/{THING_NAME}")
IOT_ENDPOINT = os.getenv("IOT_ENDPOINT", "your-iot-endpoint-ats.iot.us-east-1.amazonaws.com")

# Greengrass/Core device certs (paths on the core)
CERT_PATH = os.getenv("IOT_CERT_PATH", "/greengrass/v2/thingCert.crt")
KEY_PATH = os.getenv("IOT_KEY_PATH", "/greengrass/v2/privKey.key")
CA_PATH = os.getenv("IOT_CA_PATH", "/greengrass/v2/rootCA.pem")

# Queues
SQS_REQUEST_QUEUE_URL = os.getenv("SQS_REQUEST_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/your-req-queue")
SQS_RESPONSE_QUEUE_URL = os.getenv("SQS_RESPONSE_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/your-resp-queue")

# Working directory for temp files
TMP_DIR = os.getenv("FD_TMP_DIR", "/tmp/faces")
# -------------------------------------------------------------

sqs = boto3.client("sqs", region_name=REGION)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FaceDetection:
    def __init__(self):
        # CPU-only MTCNN is fine on edge
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def detect_face_to_file(self, image_bytes, output_dir) -> str | None:
        """
        Runs face detection on image bytes and writes a normalized RGB face crop to output_dir.
        Returns the saved file path or None if no face.
        """
        img = Image.open(image_bytes).convert("RGB")
        img_np = np.array(img)
        img_pil = Image.fromarray(img_np)

        face_tensor, prob = self.mtcnn(img_pil, return_prob=True, save_path=None)
        if face_tensor is None:
            return None

        os.makedirs(output_dir, exist_ok=True)
        # Normalize to [0, 255] and convert to PIL RGB
        face_img = face_tensor - face_tensor.min()
        denom = face_img.max() - face_img.min()
        face_img = (face_img / denom * 255) if denom > 0 else face_img * 0
        face_pil = Image.fromarray(face_img.byte().permute(1, 2, 0).numpy(), mode="RGB")

        file_path = os.path.join(output_dir, "detected_face.jpg")
        face_pil.save(file_path)
        return file_path

fd = FaceDetection()

def on_message_received(topic, payload, **kwargs):
    logging.info(f"MQTT message on '{topic}'")

    try:
        message = json.loads(payload.decode("utf-8"))
        encoded = message.get("encoded")
        request_id = message.get("request_id")
        filename = message.get("filename")

        if not encoded or not request_id or not filename:
            logging.warning("Invalid payload; required keys: encoded, request_id, filename")
            return

        img_bytes = BytesIO(base64.b64decode(encoded))
        os.makedirs(TMP_DIR, exist_ok=True)
        detected_face_path = fd.detect_face_to_file(img_bytes, TMP_DIR)

        if detected_face_path is None:
            # Optional "bonus" behavior: short-circuit when no face
            logging.info("No face detected. Sending 'No-Face' to response queue.")
            sqs.send_message(
                QueueUrl=SQS_RESPONSE_QUEUE_URL,
                MessageBody=json.dumps({
                    "request_id": request_id,
                    "filename": filename,
                    "result": "No-Face"
                })
            )
            return

        with open(detected_face_path, "rb") as fh:
            encoded_face = base64.b64encode(fh.read()).decode("utf-8")

        sqs_payload = {
            "request_id": request_id,
            "filename": filename,
            "face": encoded_face
        }
        sqs.send_message(QueueUrl=SQS_REQUEST_QUEUE_URL, MessageBody=json.dumps(sqs_payload))
        logging.info("Face detected; enqueued to request queue.")
    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)

def main():
    # Build mutual-TLS connection using Greengrass-provided certs
    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=IOT_ENDPOINT,
        cert_filepath=CERT_PATH,
        pri_key_filepath=KEY_PATH,
        ca_filepath=CA_PATH,
        client_id=THING_NAME,
        clean_session=False,
        keep_alive_secs=30,
    )

    logging.info(f"Connecting to IoT endpoint: {IOT_ENDPOINT}")
    mqtt_connection.connect().result()
    logging.info(f"Connected. Subscribing to topic: {TOPIC}")

    mqtt_connection.subscribe(topic=TOPIC, qos=mqtt.QoS.AT_LEAST_ONCE, callback=on_message_received)
    logging.info("Subscription successful.")

    try:
        threading.Event().wait()  # block forever
    except KeyboardInterrupt:
        logging.info("Disconnecting MQTT.")
        mqtt_connection.disconnect().result()

if __name__ == "__main__":
    main()
