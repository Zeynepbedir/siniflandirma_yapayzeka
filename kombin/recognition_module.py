#!/usr/bin/env python
# coding: utf-8

# In[1]:


#for read and show images
import matplotlib.pyplot as plt
import cv2                                                          
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image




# In[1]:


import os
print(os.getcwd())  # Çalışma dizinini kontrol edin


# In[2]:


import numpy as np
import random


# In[3]:


#for color classification
import colorsys                                                     
import PIL.Image as Image


# In[4]:


from scipy.spatial import KDTree
from webcolors import (
   hex_to_name,
    hex_to_rgb
)


# In[5]:


import tensorflow as tf


# In[6]:


# load pre-trained models
# please change them to your local path when load
sub_model = tf.keras.models.load_model('C:/Users/zeyne/vscodeProjeler/kombin/models/model_sub.h5')
top_model = tf.keras.models.load_model('C:/Users/zeyne/vscodeProjeler/kombin/models/model_top.h5')
bottom_model = tf.keras.models.load_model('C:/Users/zeyne/vscodeProjeler/kombin/models/model_bottom.h5')
foot_model = tf.keras.models.load_model('C:/Users/zeyne/vscodeProjeler/kombin/models/model_shoes.h5')

# In[7]:


#sub_model.summary()


# In[8]:


# all output possibilities of the model for subsequent matching
sub_list = ["bottom","foot","top"]
top_list = [['Belts', 'Blazers', 'Dresses', 'Dupatta', 'Jackets', 'Kurtas',
       'Kurtis', 'Lehenga Choli', 'Nehru Jackets', 'Rain Jacket',
       'Rompers', 'Shirts', 'Shrug', 'Suspenders', 'Sweaters',
       'Sweatshirts', 'Tops', 'Tshirts', 'Tunics', 'Waistcoat'],
           ['Boys', 'Girls', 'Men', 'Unisex', 'Women'],
           ['Black', 'Blue', 'Dark Blue', 'Dark Green', 'Dark Yellow', 'Green',
       'Grey', 'Light Blue', 'Multi', 'Orange', 'Pink', 'Purple', 'Red',
       'White', 'Yellow'],
           ['Fall', 'Spring', 'Summer', 'Winter'],
           ['Casual', 'Ethnic', 'Formal', 'Party', 'Smart Casual', 'Sports',
       'Travel']]
bottom_list = [['Capris', 'Churidar', 'Jeans', 'Jeggings', 'Leggings', 'Patiala',
       'Salwar', 'Salwar and Dupatta', 'Shorts', 'Skirts', 'Stockings',
       'Swimwear', 'Tights', 'Track Pants', 'Tracksuits', 'Trousers'],
              ['Boys', 'Girls', 'Men', 'Unisex', 'Women'],
              ['Black', 'Blue', 'Dark Blue', 'Dark Green', 'Dark Yellow', 'Grey',
       'Light Blue', 'Multi', 'Orange', 'Pink', 'Purple', 'Red', 'White',
       'Yellow'],
              ['Fall', 'Spring', 'Summer', 'Winter'],
              ['Casual', 'Ethnic', 'Formal', 'Smart Casual', 'Sports']]
foot_list = [['Casual Shoes', 'Flats', 'Flip Flops', 'Formal Shoes', 'Heels',
       'Sandals', 'Sports Sandals', 'Sports Shoes'],
            ['Boys', 'Girls', 'Men', 'Unisex', 'Women'],
            ['Black', 'Blue', 'Dark Blue', 'Dark Green', 'Dark Orange',
       'Dark Yellow', 'Grey', 'Light Blue', 'Multi', 'Orange', 'Pink',
       'Purple', 'Red', 'White', 'Yellow'],
            ['Fall', 'Spring', 'Summer', 'Winter'],
            ['Casual', 'Ethnic', 'Formal', 'Party', 'Smart Casual', 'Sports']]


# In[9]:


from scipy.spatial import KDTree  # Hızlı en yakın komşu sorguları için KDTree modülü
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb  # CSS3 renk isimlerini ve HEX-RGB dönüşümünü sağlayan modül

def convert_rgb_to_names(rgb_tuple):
    """
    Bir RGB tuple'ını CSS3 renk isimlerinden en yakın olana dönüştürür.

    Parametreler:
        rgb_tuple (tuple): RGB değerini içeren bir tuple. Örneğin, (255, 255, 255).

    Döndürür:
        str: En yakın CSS3 renk ismi. Örneğin, 'white', 'red' gibi.
    """
    # CSS3 renk veritabanını yükle (HEX kodlarını CSS3 renk isimlerine eşler)
    css3_db = CSS3_HEX_TO_NAMES

    # Renk isimleri ve RGB değerleri için listeler oluştur
    names = []  # CSS3 renk isimlerini saklayacak liste
    rgb_values = []  # HEX kodlarının RGB tuple'ına çevrilmiş değerlerini saklayacak liste

    # CSS3 renk veritabanını döngüyle işle
    for color_hex, color_name in css3_db.items():
        # Renk isimlerini listeye ekle
        names.append(color_name)
        # HEX kodlarını RGB formatına çevirerek listeye ekle
        rgb_values.append(hex_to_rgb(color_hex))

    # RGB değerleri için bir KDTree oluştur
    kdt_db = KDTree(rgb_values)  # KDTree, hızlı bir şekilde en yakın rengi bulmamıza yardımcı olur
    
    # Girdi olarak verilen RGB değeriyle KDTree'den en yakın eşleşmeyi bul
    distance, index = kdt_db.query(rgb_tuple)
    # En yakın renk ismini döndür
    return names[index]


# In[10]:


# Örnek RGB tuple
#input_rgb = (240, 248, 255)

# Renk adını bulma
#closest_color = convert_rgb_to_names(input_rgb)

# Sonucu yazdırma
#print(f"RGB {input_rgb} değerine en yakın CSS3 renk adı: '{closest_color}'.")


# In[11]:


from PIL import Image
import colorsys

def get_cloth_color(image):
    """
    Bir görüntünün baskın rengini tespit eder ve İngilizce renk adı olarak döndürür.

    Parametreler:
        image (PIL.Image): Girdi görüntüsü.

    Döndürür:
        str: Görüntünün baskın renginin İngilizce adı.
    """
    max_score = 0.0001  # En yüksek skor için başlangıç değeri
    dominant_color = None  # Baskın renk bilgisini saklayacak değişken

    # Görüntüdeki tüm renkleri analiz et
    for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):
        # RGB -> HSV dönüşümü yaparak doygunluk (saturation) hesapla
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

        # Parlaklık (brightness/luminance) kontrolü
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        y = (y - 16.0) / (235 - 16)
        if y > 0.9:  # Çok parlak renkleri atla
            continue

        # Skoru hesapla: (doygunluk + sabit bir düzeltme) * renk sayısı
        score = (saturation + 0.1) * count
        if score > max_score:  # Eğer skor önceki maksimum skoru geçerse
            max_score = score  # Maksimum skoru güncelle
            dominant_color = (r, g, b)  # Bu rengi baskın renk olarak ayarla

    # Baskın RGB rengini CSS3 renk adına dönüştür ve döndür
    return convert_rgb_to_names(dominant_color)


# In[28]:


def color_classification(single_path):
    """
    Verilen bir resim dosyasının baskın rengini sınıflandırır.

    Parametreler:
        single_path (str): Resim dosyasının bilgisayar üzerindeki yolu.

    Döndürür:
        str: Resmin baskın rengi (İngilizce renk adı).
    """
    # Resmi yükle
    image = Image.open(single_path)
    # Görüntüyü RGB formatına dönüştür
    image = image.convert('RGB')
    # Baskın rengi bul ve döndür
    return get_cloth_color(image)


# In[29]:


# Resim dosyasının yolu
#image_path = "C:/Users/zeyne/Downloads/ornek.jpg"  # Analiz edilecek resim dosyasını burada belirtin

# Fonksiyonu çalıştır
#dominant_color = color_classification(image_path)

# Sonucu yazdır
#print(f"Resimdeki baskın renk: {dominant_color}")


# In[14]:


def single_helper(train_images, my_model, lelist):
    """
    Önceden eğitilmiş bir modeli kullanarak tahmin yapar ve sonuçları anlamlı bir formatta döndürür.

    Parametreler:
        train_images: Tahmin yapılacak görüntülerin bulunduğu veri.
        my_model: Önceden eğitilmiş model (birden fazla alt-model içerir).
        lelist: Kodlayıcı listesi. Her alt-modelin tahmin ettiği sınıf etiketlerini açıklayıcı metinlere dönüştürmek için kullanılır.

    Döndürür:
        result: Modelin tahmin ettiği sonuçlardan oluşan bir liste. Her bir sonuç açıklayıcı metin formatında döndürülür.
    """
    # Model tahminlerini hesapla
    my_predictions = my_model.predict(train_images)

    # Tahmin sonuçlarını saklayacağımız bir liste
    result = []

    # Her alt-model için tahmin edilen sınıfı bul ve listeye ekle
    for i in range(len(lelist)):
        # En yüksek olasılığa sahip sınıfı bul
        type_predicted_label = np.argmax(my_predictions[i][0])
        # Sınıfı açıklayıcı metne çevir ve listeye ekle
        result.append(lelist[i][type_predicted_label])

    return result


# In[ ]:

from tensorflow.keras.preprocessing import image

def single_classification(single_path):
    """
    Bu fonksiyon bir giysi fotoğrafının yolunu alır, modeli çalıştırmak için resmi yeniden boyutlandırır
    ve sınıflandırma yapar. Sonuç olarak, alt modele yönlendirme yapmak için bir tür, giysi bilgilerini içeren bir metin ve bir liste döndürür.
    
    Girdi:
        single_path (str): Fotoğrafın dosya yolu (bu durumda resmin bellekte tutulduğu yer).

    Çıktı:
        tuple: Alt modelin türü (doğru alt modele yönlendirme için), giysi bilgilerini içeren bir metin ve detaylı bir liste.
    """
    
    # Model yalnızca veri çerçevelerine uygulanabilir.
    # Bu nedenle, modelin tek bir fotoğrafı tahmin edebilmesini sağlamak için
    # bu fotoğrafı yalnızca bir satır içeren bir veri çerçevesine dönüştürüyoruz.
    train_images = np.zeros((1,80,60,3))

    # Resmi bellekteki yol ile oku
    img = cv2.imread(single_path)  # Resmi oku
    
    # Resmi modele uygun boyuta yeniden şekillendir
    if img.shape != (80,60,3):
        img = image.load_img(single_path, target_size=(80,60,3))

    train_images[0] = img

    # İlk modelin sonuçlarına göre, doğru alt modele yönlendirme yap
    result2 = sub_list[np.argmax(sub_model.predict(train_images))]
    
    # İlk modelin sonucuna göre üç alt modele dallanır
    if result2 == "top":  # Eğer sonuç "üst" ise
        res = single_helper(train_images, top_model, top_list)
    elif result2 == "bottom":  # Eğer sonuç "alt" ise
        res = single_helper(train_images, bottom_model, bottom_list)
    elif result2 == "foot":  # Eğer sonuç "ayakkabı" ise
        res = single_helper(train_images, foot_model, foot_list)

    # Sonucu listeye ekle ve bilgileri metin formatına dönüştür
    res.append(single_path)
    res_str = f"{res[0]}, {res[1]}, {color_classification(single_path)}, {res[3]}, {res[4]}, {single_path}" 
    
    # Alt model türü, bilgi metni ve detaylı listeyi döndür
    return (result2, res_str, res)




# In[ ]:


# Giysi öneri sistemimizin faktörlerinden biri mevsimdir.
# Bu nedenle, mevcut gerçek mevsimi belirler ve uygulamada saklanan tüm giysilerle eşleştiririz.
# Bu işlem, yalnızca mevcut mevsime uygun giysilerin önerilmesini sağlar.

from datetime import date  # Tarih bilgisi almak için gerekli modül içe aktarılıyor

# Bugünün tarihini alıyoruz
todays_date = date.today()

# Bugünün tarihinden ay bilgisini çıkarıyoruz
tomonth = todays_date.month

# Ay bilgisine göre mevsimi belirliyoruz
if tomonth in [3, 4, 5]:  # Mart, Nisan, Mayıs ayları
    toseason = "Spring"   # İlkbahar
elif tomonth in [6, 7, 8]:  # Haziran, Temmuz, Ağustos ayları
    toseason = "Summer"   # Yaz
elif tomonth in [9, 10, 11]:  # Eylül, Ekim, Kasım ayları
    toseason = "Fall"     # Sonbahar
else:  # Geriye kalan aylar: Aralık, Ocak, Şubat
    toseason = "Winter"   # Kış

