import os
import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis

# Inisialisasi InsightFace untuk analisis wajah
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

# Ekstrak pose (angle) dari satu gambar
def extract_pose_single(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gagal baca gambar: {image_path}")
        return None

    faces = face_app.get(img)
    if not faces:
        print(f"❌ Tidak ada wajah: {image_path}")
        return None

    face = faces[0]
    keypoints = face.kps

    # Estimasi pose dari posisi mata
    left_eye = keypoints[0]
    right_eye = keypoints[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    return {
        "angle": float(angle),
        "bbox": face.bbox.tolist()
    }

# Ekstrak pose dari semua gambar dalam folder dan simpan sebagai JSON
def extract_poses_from_folder(folder_path, output_path):
    result = {}
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".jpg"):
            continue
        path = os.path.join(folder_path, fname)
        pose = extract_pose_single(path)
        if pose:
            result[fname] = pose

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
