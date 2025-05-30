import json

def merge_coco_jsons(json_paths, output_path):
    merged = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    annotation_id_offset = 0
    image_id_offset = 0

    for i, path in enumerate(json_paths):
        with open(path, 'r') as f:
            data = json.load(f)

            # 카테고리는 첫 json에서만 복사
            if i == 0:
                merged["categories"] = data["categories"]

            # 이미지, 어노테이션에 id 중복방지 처리 필요
            for img in data["images"]:
                img["id"] += image_id_offset
                merged["images"].append(img)

            for ann in data["annotations"]:
                ann["id"] += annotation_id_offset
                ann["image_id"] += image_id_offset
                merged["annotations"].append(ann)

            # id offset 업데이트 (max id + 1)
            image_id_offset = max(img["id"] for img in merged["images"]) + 1
            annotation_id_offset = max(ann["id"] for ann in merged["annotations"]) + 1

    with open(output_path, 'w') as f:
        json.dump(merged, f)

merge_coco_jsons([
    "COCO Subset.v4-80-15-5-ratio-with-5-classes.coco/train/_annotations.coco.json",
    "COCO Subset.v4-80-15-5-ratio-with-5-classes.coco/valid/_annotations.coco.json",
    "COCO Subset.v4-80-15-5-ratio-with-5-classes.coco/test/_annotations.coco.json"
], "COCO Subset.v4-80-15-5-ratio-with-5-classes.coco/_annotations.coco.json")
