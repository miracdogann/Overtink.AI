import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Dataset Yolu
dataset_dir = "dataset"

# 2. Parametreler
batch_size = 16
img_height = 224
img_width = 224
epochs = 50
seed = 42

# 3. Veri setini yükle (train ve validation olarak)
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

class_names = train_ds.class_names
print("Sınıflar:", class_names)

# 4. Veri ön işleme: Ölçeklendirme
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 5. Performansı artırmak için dataset önbellekleme ve prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 6. Model Oluşturma: Basit CNN veya Transfer Learning
# Burada Transfer Learning - MobileNetV2 kullanıyorum
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False  # Önceden eğitilmiş ağırlıkları dondur

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 7. Model Eğitimi
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Sınıf isimlerini de kaydet (json olarak)
import json
with open("class_names1.json", "w") as f:
    json.dump(class_names, f)

# model.save("ped2.h5")
# 8. Modeli değerlendir
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# 9. Tahmin ve detaylı analiz
# Validation datasını al
val_images = []
val_labels = []

for images, labels in val_ds:
    val_images.append(images.numpy())
    val_labels.append(labels.numpy())

val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)

pred_probs = model.predict(val_images)
pred_labels = np.argmax(pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(val_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

unique_val_labels = np.unique(val_labels)
subset_class_names = [class_names[i] for i in unique_val_labels]
# Classification report'u sadece mevcut sınıflar için üret
report = classification_report(
    val_labels,
    pred_labels,
    labels=unique_val_labels,
    target_names=subset_class_names,
    output_dict=True
)
print("Classification Report:")
print(classification_report(val_labels, pred_labels, labels=unique_val_labels, target_names=subset_class_names))
# Hata oranı = 1 - doğruluk
error_rate = 1 - val_acc
print(f"Hata Oranı: {error_rate*100:.2f}%")

# 10. Örnek bir fotoğrafla test et (örnek yol ver)
from tensorflow.keras.preprocessing import image

def test_single_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    print(f"Test edilen resim: {img_path}")
    print(f"Tahmin edilen sınıf: {pred_class} (%{confidence:.2f} doğrulukla)")

# Örnek test
test_single_image("C:/Users/mirac/Desktop/hackhaton/yarisma/yapay_zeka/dataset/hatasiz/aci77.jpg")
