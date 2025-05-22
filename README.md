# Medikal Ürün Kalite Kontrolü Hackathon Projesi

Bu proje, **TURGUTLU TEKNOLOJİ GÜNLERİ - 5 Görüntü İşleme ile Akıllı Üretim: Medikal Ürün Kalite Kontrolü Hackathonu** için geliştirilmiştir.

## 🚩 #ttg5hackathon2025 

## ⚙️ Backend Yapılandırılması
backend/ped_app dizinine geçerek 
terminalde pip install -r requirements.txt  (komutu çalıştırılmalıdır) 
daha sonra yine yanı dizinde python manage.py runserver yaparak backend tarafı çalıştırılmalıdır 


 ------- Frontend Yapılandırılması ----- 
 Node.js npm yüklü olmalı , andorid stuido indirilip emülator desteği sağlanmalıdır 
frontend/pedapp dizinine geçerek 
terminalde npm install komutu çalıştırılarak gerekli bağımlılıkların yüklenmesi sağlanmalıdır
daha sonra npx expo start diyerek proje çalıştırılabilir 

Ayrıca bu projede emeği geçen ekip arkadaşlarıma teşşekür ederim 
Miraç Doğan (Yazılım Mühendisliği)
Yağız Belyan Çelik (Yazılım Mühendisliği)
Mert Sezginer (Mekatronik Mühendisliği)

## 📊 Karmaşıklık Analizi (Big-O Notasyonu)

Bu projede Transfer Learning yöntemiyle `MobileNetV2` modeli kullanılmış ve TensorFlow üzerinden bir görüntü sınıflandırma pipeline'ı kurulmuştur. Aşağıda modelin ve işlemlerin teorik zaman ve mekân karmaşıklıkları verilmiştir.

---

### 🧠 Model Eğitimi (`model.fit`)

- **Zaman Karmaşıklığı:** `O(n * e)`  
  - `n`: Eğitim örneği sayısı  
  - `e`: Epoch sayısı  
  - Not: MobileNetV2 katmanları dondurulduğu için sabit maliyetlidir.
  
- **Mekân Karmaşıklığı:** `O(p)`  
  - `p`: Toplam model parametre sayısı  
  - MobilNetV2 yaklaşık 2.2M parametre içerir.

---

### 🖼️ Tek Görüntü Tahmini (`test_single_image()`)

- **Zaman Karmaşıklığı:** `O(1)`  
  - Sabit boyuttaki (224x224) bir görüntü üzerinden tahmin yapılır.

- **Mekân Karmaşıklığı:** `O(1)`  
  - Bellekte sadece bir görüntü tutulur ve model zaten yüklenmiş durumdadır.

---

### 🗂️ Veri Seti Yükleme ve Ön İşleme

- **Zaman Karmaşıklığı:** `O(n)`  
  - `n`: Toplam görüntü sayısı  
  - Her görsel tek tek yüklenir ve yeniden boyutlandırılır (rescale işlemi).

- **Mekân Karmaşıklığı:** `O(b)`  
  - `b`: Batch size (örn: 16) kadar görüntü bellekte tutulur.  
  - Prefetch ve cache sayesinde verimli kullanım sağlanır.

---

### 📊 Değerlendirme Aşaması

> Kullanılan fonksiyonlar: `confusion_matrix`, `classification_report`

- **Zaman Karmaşıklığı:** `O(n)`  
  - `n`: Doğrulama veri sayısı kadar tahmin yapılır ve kıyaslanır.

- **Mekân Karmaşıklığı:** `O(n)`  
  - Tahmin etiketleri, gerçek etiketler ve rapor çıktıları bellekte tutulur.

---

### 📌 Özet Tablo

| İşlem                         | Zaman Karmaşıklığı | Mekân Karmaşıklığı |
|------------------------------|---------------------|---------------------|
| Model Eğitimi                | O(n * e)            | O(p)                |
| Tek Görüntü Tahmini          | O(1)                | O(1)                |
| Dataset Yükleme & Rescaling  | O(n)                | O(b)                |
| Değerlendirme (Doğrulama)    | O(n)                | O(n)                |




