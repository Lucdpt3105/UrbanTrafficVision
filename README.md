# UrbanTrafficVision

DNN-based vehicle detection system using YOLOv8

## Setup & Installation

### 1. Clone repository
```bash
git clone <repo-url>
cd DNN-Vehicle
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv8 models
Chạy lệnh sau để tự động download models:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8m.pt')"
```

Hoặc chạy script:
```bash
python -c "
from ultralytics import YOLO
print('Downloading yolov8n...')
YOLO('yolov8n.pt')
print('Downloading yolov8m...')
YOLO('yolov8m.pt')
print('Done!')
"
```

## Usage

```bash
python detect.py
```