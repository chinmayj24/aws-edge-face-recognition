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

REGION = 'us-east-1'
THING_NAME = '1230031818-IoTThing'
TOPIC = f'clients/{THING_NAME}'
SQS_REQUEST_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/619071330649/1230031818-req-queue'
SQS_RESPONSE_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/619071330649/1230031818-resp-queue'
IOT_ENDPOINT = 'ap9hdxkvj2x4s-ats.iot.us-east-1.amazonaws.com'

sqs = boto3.client('sqs', region_name=REGION)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,  
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class face_detection:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def face_detection_func(self, image_bytes, output_path):
        img = Image.open(image_bytes).convert("RGB")
        img = np.array(img)
        img = Image.fromarray(img)

        face, prob = self.mtcnn(img, return_prob=True, save_path=None)

        if face is not None:
            os.makedirs(output_path, exist_ok=True)
            face_img = face - face.min()
            face_img = (face_img / face_img.max()) * 255
            face_pil = Image.fromarray(face_img.byte().permute(1, 2, 0).numpy())
            file_path = os.path.join(output_path, "detected_face.jpg")
            face_pil.save(file_path)
            return file_path
        else:
            return None

fd = face_detection()

def on_message_received(topic, payload, **kwargs):
    logging.info(f"Received MQTT message on topic {topic}")

    try:
        message_image = json.loads(payload.decode('utf-8'))
        encoded = message_image.get('encoded')
        request_id = message_image.get('request_id')
        filename = message_image.get('filename')

        if not encoded or not request_id or not filename:
            logging.warning("Invalid payload format.")
            return

        img_bytes = BytesIO(base64.b64decode(encoded))
        temp_dir = '/tmp/faces'
        os.makedirs(temp_dir, exist_ok=True)

        detected_face_path = fd.face_detection_func(img_bytes, temp_dir)

        if detected_face_path is None:
            logging.info("No face is detected. Sending 'No-Face' result to response queue.")
            sqs.send_message(
                QueueUrl=SQS_RESPONSE_QUEUE_URL,
                MessageBody=json.dumps({
                    'request_id': request_id,
                    'filename': filename,
                    'result': 'No-Face'
                })
            )
            return

        with open(detected_face_path, 'rb') as face_file:
            face_bytes = face_file.read()
            encoded_face = base64.b64encode(face_bytes).decode('utf-8')

        sqs_payload = {
            'request_id': request_id,
            'filename': filename,
            'face': encoded_face
        }

        sqs.send_message(
            QueueUrl=SQS_REQUEST_QUEUE_URL,
            MessageBody=json.dumps(sqs_payload)
        )
        logging.info("Face detected and message sent to request queue.")

    except Exception as e:
        logging.error(f"Error during processing is: {str(e)}")

def main():
    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=IOT_ENDPOINT,
        cert_filepath="/greengrass/v2/thingCert.crt",
        pri_key_filepath="/greengrass/v2/privKey.key",
        ca_filepath="/greengrass/v2/rootCA.pem",
        client_id=THING_NAME,
        clean_session=False,
        keep_alive_secs=30,
    )

    logging.info("Connecting to the MQTT broker")
    mqtt_connection.connect().result()
    logging.info(f"Connected to the broker. Subscribing to the {TOPIC}")

    mqtt_connection.subscribe(topic=TOPIC, qos=mqtt.QoS.AT_LEAST_ONCE, callback=on_message_received)
    logging.info("Subscription is fully successful.")

    try:
        threading.Event().wait() 
    except KeyboardInterrupt:
        logging.info("Disconnecting from MQTT.")
        mqtt_connection.disconnect().result()

if __name__ == "__main__":
    main()