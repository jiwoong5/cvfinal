import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

DATASET_DIR = Path("dataset")  # 이미지 폴더 경로

def run_yolov5_and_collect_times(dataset_dir):
    image_paths = list(dataset_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No JPG images found in {dataset_dir}")
        return []

    inference_times = []

    for img_path in image_paths:
        print(f"Processing {img_path}...")
        result = subprocess.run([
            'python', 'yolov5/detect.py',
            '--weights', 'yolov5s.pt',
            '--img', '640',
            '--conf', '0.25',
            '--source', str(img_path),
        ], capture_output=True, text=True)

        # 로그 메시지 stderr에서 추출
        log_output = result.stderr
        for line in log_output.splitlines():
            if "ms inference" in line:
                parts = line.split(',')
                for part in parts:
                    if "inference" in part:
                        time_str = part.strip().split('ms')[0]
                        try:
                            time_val = float(time_str)
                            inference_times.append(time_val)
                        except ValueError:
                            pass

    return inference_times

def plot_inference_times(times):
    plt.figure(figsize=(8, 5))
    plt.plot(times, marker='o', linestyle='-', color='blue')
    plt.title("YOLOv5 Inference Times per Image")
    plt.xlabel("Image Index")
    plt.ylabel("Inference Time (ms)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if not DATASET_DIR.exists():
        print(f"Dataset directory does not exist: {DATASET_DIR}")
    else:
        times = run_yolov5_and_collect_times(DATASET_DIR)
        if times:
            print(f"Collected {len(times)} inference times.")
            plot_inference_times(times)
        else:
            print("No inference times collected.")

