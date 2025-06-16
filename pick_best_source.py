import json
import math

def pick_best_source(target_path, source_json_path):
    from extract_pose_dataset import extract_pose_single

    # Ekstrak pose dari target
    target_pose = extract_pose_single(target_path)
    if target_pose is None:
        print("❌ Tidak bisa ekstrak pose target.")
        return None

    with open(source_json_path) as f:
        sources = json.load(f)

    best_name = None
    best_score = float("inf")

    for name, item in sources.items():
        angle_diff = abs(target_pose["angle"] - item["angle"])
        if angle_diff < best_score:
            best_score = angle_diff
            best_name = name

    return best_name
