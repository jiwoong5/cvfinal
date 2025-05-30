import subprocess
from pathlib import Path

DATASET_DIR = Path("dataset")  # 이미지 폴더 경로

def run_yolov5_on_dataset(dataset_dir):
    # dataset 폴더 내 모든 jpg 파일 리스트
    image_paths = list(dataset_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No JPG images found in {dataset_dir}")
        return
    
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        subprocess.run([
            'python', 'yolov5/detect.py',
            '--weights', 'yolov5s.pt',
            '--img', '640',
            '--conf', '0.25',
            '--source', str(img_path),
            '--save-txt', '--save-conf'
        ])

if __name__ == "__main__":
    if not DATASET_DIR.exists():
        print(f"Dataset directory does not exist: {DATASET_DIR}")
    else:
        run_yolov5_on_dataset(DATASET_DIR)

