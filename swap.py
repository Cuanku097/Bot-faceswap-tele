import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

# Load gambar sumber dan target
source_img = cv2.imread("wajah.jpg")
target_img = cv2.imread("target1.jpg")

# Inisialisasi face analysis
faceapp = FaceAnalysis(name="buffalo_l")
faceapp.prepare(ctx_id=0)

# Deteksi wajah
source_faces = faceapp.get(source_img)
target_faces = faceapp.get(target_img)

if len(source_faces) == 0 or len(target_faces) == 0:
    print("❌ Wajah tidak terdeteksi!")
    exit()

# Ambil wajah pertama
source_face = source_faces[0]
target_face = target_faces[0]

# Load model inswapper
swapper = model_zoo.get_model('/root/inswapper_128.onnx', download=False)

# Lakukan swap wajah
output_img = swapper.get(target_img, target_face, source_face)

# Simpan hasil
cv2.imwrite("hasil_swap.jpg", output_img)
print("✅ Swap selesai! File: hasil_swap.jpg")
