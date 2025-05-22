import os
from io import BytesIO
from PIL import Image
from rest_framework.decorators import api_view
from rest_framework.response import Response
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import matplotlib.pyplot as plt


# MODEL VE SINIFLAR
img_height = 224
img_width = 224
model = tf.keras.models.load_model("ped2.h5")
with open("class_names1.json", "r") as f:
    class_names = json.load(f)

def test_single_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = preds[pred_index] * 100

    class_probs = {class_names[i]: float(preds[i] * 100) for i in range(len(class_names))}

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "class_probabilities": class_probs
    }


@api_view(['POST'])
def predict_image_api(request):
    file_obj = request.FILES.get('image', None)
    
    if not file_obj:
        return Response({"error": "Resim dosyası gönderilmedi."}, status=400)

    try:
        # Gelen dosyayı bellekte aç
        img = Image.open(file_obj)
        img = img.convert('RGB')  # Eğer RGBA, L vs. ise dönüştür
        img = img.resize((img_width, img_height))

        # Görüntüyü tensöre çevir
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Tahmin yap
        preds = model.predict(img_array)[0]
        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]
        confidence = preds[pred_index] * 100

        class_probs = {class_names[i]: float(preds[i] * 100) for i in range(len(class_names))}

        return Response({
            "message": "Resim başarıyla analiz edildi.",
            "predicted_class": pred_class,
            "confidence": f"{confidence:.2f}%",
            "class_probabilities": class_probs
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)




@api_view(['GET'])
def ped_api(request):
    return Response({"message": "Hello, world!"})

