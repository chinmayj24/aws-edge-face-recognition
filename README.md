# AWS Edge Computing – IoT + Greengrass + Lambda Face Recognition

A distributed **Edge–Cloud Face Recognition system** that combines **AWS IoT Greengrass**, **MQTT**, and **AWS Lambda**.  
The project simulates IoT devices using EC2 instances to perform **edge-side face detection** and **cloud-based face recognition**, enabling low-latency, privacy-preserving machine learning at the edge.

---

## 🧠 Overview

This project demonstrates an **IoT edge computing pipeline** where:
- **IoT devices** (simulated via EC2) act as smart cameras publishing video frames.
- **AWS IoT Greengrass Core** performs **on-device face detection** using `MTCNN`.
- **AWS Lambda** performs **face recognition** using `PyTorch` (ResNet-based model).
- **Amazon SQS** connects the edge and cloud layers asynchronously.

---

## ⚙️ Architecture

<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/d3e57c96-967a-4313-aa9a-931a67a1afd1" />

### 🔹 Data Flow
1. **IoT Client Device (EC2)**
   - Publishes Base64-encoded video frames via MQTT to topic:  
     `clients/<thing-name>`  
   - Uses the **AWS IoT Device SDK v2**.

2. **Greengrass Core Device (EC2)**
   - Runs the **Face Detection Component (`fd_component.py`)**.
   - Detects faces locally using `MTCNN` (facenet-pytorch).
   - If faces found → sends to **SQS Request Queue** for recognition.  
   - If no faces detected → sends `"No-Face"` result directly to the **SQS Response Queue**.

3. **AWS Lambda Function**
   - Runs **`fr_lambda.py`** for face recognition.
   - Triggered automatically by the **SQS Request Queue**.
   - Uses a **PyTorch ResNet** model for embedding comparison.
   - Sends `{request_id, result}` back to the **SQS Response Queue**.

4. **Client**
   - Polls the **Response Queue** to retrieve recognition results.

---

## ☁️ AWS Resources Used

| Service | Example Name | Purpose |
|----------|---------------|----------|
| **EC2 (Core)** | `IoT-Greengrass-Core` | Runs AWS IoT Greengrass Core |
| **EC2 (Client)** | `IoT-Greengrass-Client` | Simulated IoT device (frame publisher) |
| **IoT Thing** | `your-iot-thing-name` | Represents the IoT device |
| **Greengrass Component** | `com.clientdevices.FaceDetection` | Edge component for face detection |
| **Lambda** | `face-recognition` | Cloud face recognition function |
| **SQS** | `your-req-queue` / `your-resp-queue` | Message queues between edge and cloud |
| **IoT Core** | MQTT endpoint | Communication between devices |
| **IAM Role** | `lambda-exec-role` | Allows Lambda and Greengrass to access SQS |

---

## 🧩 Project Structure

  ```
  .
  ├── face-detection/
  │ └── fd_component.py # Greengrass Core Component (Face Detection)
  └── face-recognition/
  └── fr_lambda.py # AWS Lambda Function (Face Recognition)
  ```


---

## ⚙️ Environment Configuration

Both the Greengrass component and Lambda function use **environment variables** —  
no hardcoded credentials or URLs are stored in code.

### 🧠 For `fd_component.py` (Greengrass component)
  ```bash
  AWS_REGION=us-east-1
  IOT_THING_NAME=your-iot-thing-name
  IOT_TOPIC=clients/your-iot-thing-name
  IOT_ENDPOINT=your-iot-endpoint-ats.iot.us-east-1.amazonaws.com
  IOT_CERT_PATH=/greengrass/v2/thingCert.crt
  IOT_KEY_PATH=/greengrass/v2/privKey.key
  IOT_CA_PATH=/greengrass/v2/rootCA.pem
  SQS_REQUEST_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/your-req-queue
  SQS_RESPONSE_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/your-resp-queue
  FD_TMP_DIR=/tmp/faces
  ```

### 🧠 For fr_lambda.py (AWS Lambda)

  ```bash
  AWS_REGION=us-east-1
  SQS_RESPONSE_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/your-resp-queue
  MODEL_PATH=resnetV1.pt
  MODEL_WT_PATH=resnetV1_video_weights.pt
  ```

## 🚀 Setup Guide (Simplified)
### 1️⃣ Launch EC2 Instances

- Core Device: Amazon Linux 2023 → installs IoT Greengrass Core.
- Client Device: Ubuntu → installs AWS IoT Device SDK.
- Allow inbound traffic from all IPs (for lab simulation only).

### 2️⃣ Install AWS IoT Greengrass Core
  ```bash
  sudo dnf install java-11-amazon-corretto -y
  cd ~
  curl -s https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip -o greengrass.zip
  unzip greengrass.zip -d GreengrassInstaller && rm greengrass.zip
  sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE \
   -jar ./GreengrassInstaller/lib/Greengrass.jar \
   --aws-region us-east-1 \
   --thing-name MyGreengrassCore \
   --thing-group-name MyGreengrassCoreGroup \
   --provision true --setup-system-service true --deploy-dev-tools true
  ```

### 3️⃣ Create and Deploy the Face Detection Component

  ```bash
  mkdir -p ~/greengrassv2/{recipes,artifacts}
  # Add recipe JSON in recipes/
  # Add fd_component.py in artifacts/com.clientdevices.FaceDetection/1.0.0/
  sudo /greengrass/v2/bin/greengrass-cli deployment create \
   --recipeDir ~/greengrassv2/recipes \
   --artifactDir ~/greengrassv2/artifacts \
   --merge "com.clientdevices.FaceDetection=1.0.0"
  ```

### 4️⃣ Deploy the Face Recognition Lambda

- Runtime: Python 3.9+
- Handler: fr_lambda.lambda_handler
- Trigger: SQS Request Queue
- Environment variables: (as shown above)
- Permissions:
    AWSLambdaSQSQueueExecutionRole, AWSLambdaVPCAccessExecutionRole

### 🔄 Data Flow Summary

  ```
  IoT Client → (MQTT topic: clients/<thing>)
       ↓
  Greengrass Core (fd_component.py)
       ↓
  SQS Request Queue
       ↓
  AWS Lambda (fr_lambda.py)
       ↓
  SQS Response Queue
       ↓
  Client retrieves recognition results

  ```

## 🧠 Tech Stack

- AWS IoT Greengrass v2
- AWS IoT Core (MQTT)
- AWS Lambda + SQS
- Python 3.9+
- facenet-pytorch, torch, numpy, Pillow, awscrt, awsiotsdk
