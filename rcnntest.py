import torch
import cv2
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import time
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESHOLD = 0.5

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def run_fasterrcnn(image_path):
    print("[Faster R-CNN] 추론 시작")

    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image_rgb).to(DEVICE)

    # 모델 로드 및 평가모드
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE)
    model.eval()

    with torch.no_grad():
        start = time.time()
        outputs = model([img_tensor])
        end = time.time()

    print(f"[Faster R-CNN] 추론 시간: {(end - start)*1000:.1f} ms")

    return outputs, image

def main():
    IMAGE_PATH = "test.jpg"

    if not Path(IMAGE_PATH).exists():
        print(f"이미지 파일이 존재하지 않습니다: {IMAGE_PATH}")
        return

    outputs, image = run_fasterrcnn(IMAGE_PATH)
    if outputs is None:
        return

if __name__ == "__main__":
    main()
