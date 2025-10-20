import os
import json
import boto3
import base64
import tempfile
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

REGION = "us-east-1"
SQS_RESPONSE_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/619071330649/1230031818-resp-queue"
sqs = boto3.client("sqs", region_name=REGION)

MODEL_PATH = "resnetV1.pt"
MODEL_WT_PATH = "resnetV1_video_weights.pt"

class face_recognition:
    def face_recognition_func(self, model_path, model_wt_path, face_img_path):
        face_pil = Image.open(face_img_path).convert("RGB")
        key = os.path.splitext(os.path.basename(face_img_path))[0].split(".")[0]

        face_numpy = np.array(face_pil, dtype=np.float32)
        face_numpy /= 255.0
        face_numpy = np.transpose(face_numpy, (2, 0, 1))
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)

        saved_data = torch.load(model_wt_path)
        self.resnet = torch.jit.load(model_path).eval()

        if face_tensor is not None:
            emb = self.resnet(face_tensor.unsqueeze(0)).detach()
            embedding_list = saved_data[0]
            name_list = saved_data[1]

            dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
            idx_min = dist_list.index(min(dist_list))
            return name_list[idx_min]
        else:
            print("No face is detected")
            return "Unknown"

recognizer = face_recognition()

def lambda_handler(event, context):
    for record in event['Records']:
        pld = json.loads(record['body'])
        request_id = pld['request_id']
        face_b64 = pld['face']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp") as tmpF:
            tmpF.write(base64.b64decode(face_b64))
            face_img_path = tmpF.name

        labelpred = recognizer.face_recognition_func(
            model_path=MODEL_PATH,
            model_wt_path=MODEL_WT_PATH,
            face_img_path=face_img_path
        )

        result_msg = {
            "request_id": request_id,
            "result": labelpred
        }

        sqs.send_message(
            QueueUrl=SQS_RESPONSE_QUEUE_URL,
            MessageBody=json.dumps(result_msg)
        )

        os.remove(face_img_path)

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Face recognition completed."})
    }
