import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import matplotlib.pyplot as plt

# Parametreler
img_height = 224
img_width = 224

# Modeli yükle
model = tf.keras.models.load_model("ped2.h5")

# Sınıf isimlerini yükle
with open("class_names1.json", "r") as f:
    class_names = json.load(f)

def test_single_image(img_path):
    # Görüntüyü hazırla
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin yap
    preds = model.predict(img_array)[0]  # tek resim olduğu için [0] alıyoruz
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = preds[pred_index] * 100

    # Sonucu yazdır
    print(f"Test edilen resim: {img_path}")
    print(f"Tahmin edilen sınıf: {pred_class}")
    print(f"Doğruluk: %{confidence:.2f}")
    print("\n--- Tüm sınıf olasılıkları ---")
    for i, prob in enumerate(preds):
        print(f"{class_names[i]}: %{prob * 100:.2f}")

    # Bar grafiği olarak göster
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, preds * 100, color='skyblue')
    plt.title(f"'{pred_class}' sınıfı %{confidence:.2f} olasılıkla tahmin edildi")
    plt.ylabel("Olasılık (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Örnek test
test_single_image("C:/Users/mirac/Desktop/hackhaton/yarisma/yapay_zeka/dataset/yapisal_bozulma/bozulma3.jpg")
