import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from pick_best_source import pick_best_source

# 🔧 Konfigurasi
TOKEN = "7677353768:AAEUPMwBeDM6X-s-pboXU7FrcLbX9K8jOHs"
SOURCE1_DIR = "source1"
SOURCE2_DIR = "source2"
OUTPUT_DIR = "hasil"

# 📂 Inisialisasi direktori
os.makedirs(SOURCE1_DIR, exist_ok=True)
os.makedirs(SOURCE2_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🧠 Inisialisasi InsightFace
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)
swapper = get_model("/root/inswapper_128.onnx", download=False)

# 🧹 Bersihkan file invalid
def clean_invalid_sources(source_dir):
    removed = []
    for fname in os.listdir(source_dir):
        fpath = os.path.join(source_dir, fname)
        try:
            if not fname.lower().endswith(".jpg"):
                continue
            if os.path.getsize(fpath) == 0 or cv2.imread(fpath) is None:
                os.remove(fpath)
                removed.append(fname)
        except:
            try:
                os.remove(fpath)
                removed.append(fname)
            except:
                pass
    return removed

# 📥 Simpan gambar (foto atau dokumen)
def save_image_from_message(message):
    try:
        if message.photo:
            file = message.photo[-1].get_file()
        elif message.document and message.document.mime_type.startswith("image/"):
            file = message.document.get_file()
        else:
            return None
        file_bytes = BytesIO()
        file.download(out=file_bytes)
        file_bytes.seek(0)
        img = Image.open(file_bytes).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ERROR] Gagal membaca gambar: {e}")
        return None

# 📸 Handler utama
def handle_photo(update: Update, context: CallbackContext):
    msg = update.message
    caption = (msg.caption or "").strip().lower()

    img = save_image_from_message(msg)
    if img is None:
        msg.reply_text("❌ Gagal membaca gambar. Kirim sebagai *foto* atau *dokumen gambar*.")
        return

    faces = face_app.get(img)
    if not faces:
        msg.reply_text("❌ Wajah tidak terdeteksi.")
        return

    # 🔁 Reset source
    if caption == "-reset1":
        for f in os.listdir(SOURCE1_DIR):
            os.remove(os.path.join(SOURCE1_DIR, f))
        msg.reply_text("✅ Source 1 direset.")
        return
    if caption == "-reset2":
        for f in os.listdir(SOURCE2_DIR):
            os.remove(os.path.join(SOURCE2_DIR, f))
        msg.reply_text("✅ Source 2 direset.")
        return

    # ➕ Tambah source
    if caption == "-s":
        path = os.path.join(SOURCE1_DIR, f"{msg.message_id}.jpg")
        cv2.imwrite(path, img)
        msg.reply_text("✅ Wajah ditambahkan ke Source 1.")
        return
    if caption == "-s2":
        path = os.path.join(SOURCE2_DIR, f"{msg.message_id}.jpg")
        cv2.imwrite(path, img)
        msg.reply_text("✅ Wajah ditambahkan ke Source 2.")
        return

    # 🤖 Pilih source aktif
    use_source2 = (caption == "-2")
    source_dir = SOURCE2_DIR if use_source2 else SOURCE1_DIR
    poses_path = f"/root/faceswap/{source_dir}_faces.json"
    os.makedirs(source_dir, exist_ok=True)

    removed = clean_invalid_sources(source_dir)
    if removed:
        msg.reply_text(f"⚠️ {len(removed)} file source rusak dihapus otomatis.")

    if not os.listdir(source_dir):
        msg.reply_text("❌ Source kosong.")
        return

    # Simpan target sementara
    temp_target_path = "/root/faceswap/target.jpg"
    cv2.imwrite(temp_target_path, img)

    # 🧠 Buat ulang JSON jika belum ada
    if not os.path.exists(poses_path):
        msg.reply_text("ℹ️ Membuat database wajah dari source...")
        from subprocess import run
        run(["python3", "pick_best_source.py", source_dir, poses_path])

    # Pilih wajah terbaik
    best_source_name = pick_best_source(temp_target_path, poses_path)
    if not best_source_name:
        msg.reply_text("❌ Tidak bisa memilih wajah source yang cocok.")
        return

    best_source_path = os.path.join(source_dir, best_source_name)
    src_img = cv2.imread(best_source_path)
    if src_img is None:
        msg.reply_text("❌ File source tidak bisa dibaca.")
        return

    src_faces = face_app.get(src_img)
    if not src_faces:
        msg.reply_text("❌ Wajah source tidak terdeteksi.")
        return

    best_face = src_faces[0]

    # 🔄 Swap wajah
    for face in faces:
        img = swapper.get(img, face, best_face, paste_back=True)

    # 💾 Simpan & kirim hasil
    out_path = os.path.join(OUTPUT_DIR, f"{msg.message_id}_swap.jpg")
    cv2.imwrite(out_path, img)
    with open(out_path, "rb") as f:
        msg.reply_photo(photo=f, caption=f"✅ Ganti wajah pakai {best_source_name}")

# 🚀 Jalankan bot
def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.photo | Filters.document.image, handle_photo))
    updater.start_polling()
    print("🤖 Bot aktif...")
    updater.idle()

if __name__ == "__main__":
    main()
