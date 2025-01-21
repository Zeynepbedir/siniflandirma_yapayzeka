from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2                                                          
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
import os
import random
import colorsys                                                     
#import PIL.Image as Image
from PIL import Image
from scipy.spatial import KDTree
from webcolors import (
   hex_to_name,
    hex_to_rgb
)
import random
from flask_cors import CORS
import io  #veri akışı için

from recognition_module import *

# Modelleri yükleme
sub_model = tf.keras.models.load_model('C:/Users/zeyne/vscodeProjeler/kombin/models/model_sub.h5')
top_model = tf.keras.models.load_model('C:/Users/zeyne/vscodeProjeler/kombin/models/model_top.h5')
bottom_model = tf.keras.models.load_model('C:/Users/zeyne/vscodeProjeler/kombin/models/model_bottom.h5')
foot_model = tf.keras.models.load_model('C:/Users/zeyne/vscodeProjeler/kombin/models/model_shoes.h5')

# Flask uygulamasını başlatma
app = Flask(__name__)


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

# API endpoint'ini tanımlıyoruz
@app.route('/convert_rgb_to_name', methods=['POST'])
def api_convert_rgb_to_name():
    """
    JSON formatında RGB değeri alır ve bu değeri CSS3 renk ismine dönüştürür.
    """
    try:
        # İstekten gelen JSON verisini alıyoruz
        data = request.get_json()       
        # RGB tuple'ını alıyoruz
        rgb_tuple = tuple(data.get('rgb'))  # 'rgb' parametresi (örneğin [255, 255, 255]) şeklinde     
        # Fonksiyonu çağırarak renk ismini alıyoruz
        color_name = convert_rgb_to_names(rgb_tuple)       
        # Sonucu JSON formatında döndürüyoruz
        return jsonify({'color_name': color_name}), 200
        
    except Exception as e:
        # Hata durumunda 400 hata kodu ve mesajı döndürüyoruz
        return jsonify({'error': str(e)}), 400

@app.route('/get_dominant_color', methods=['POST'])
def get_dominant_color():
    """
    Kullanıcıdan gelen bir resmin baskın rengini tespit eder ve rengin İngilizce adını döndürür.
    """
    try:
        # Kullanıcıdan gelen resim verisini al
        img_file = request.files['image']
        image = Image.open(img_file)

        # get_cloth_color fonksiyonunu çağırarak baskın rengi tespit et
        dominant_color = get_cloth_color(image)

        # Baskın rengin İngilizce ismini döndür
        return jsonify({'dominant_color': dominant_color})

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/color_classification', methods=['POST'])
def color_classification_api():
    """
    Kullanıcıdan gelen bir resmin baskın rengini sınıflandırır.
    Resmin dosya yolunu alır ve rengini döndürür.
    """
    try:
        # Kullanıcıdan gelen resim dosyasını al
        img_file = request.files['image']
        
         # Resmi belleğe al
        image = Image.open(img_file.stream) 
        
        # color_classification fonksiyonunu çağırarak baskın rengi tespit et
        dominant_color = color_classification(img_path)

        # Baskın rengin İngilizce ismini döndür
        return jsonify({'dominant_color': dominant_color})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/single_helper', methods=['POST'])
def single_helper_api():
    try:
        # API'ye gelen veriyi al
        data = request.get_json()

        # Gerekli parametreleri al
        train_images = np.array(data['train_images'])  # Görselleri al
        lelist = data['lelist']  # Kodlayıcı listesi
        #my_model = tf.keras.models.load_model('path_to_your_model')  # Eğitilmiş modelin yolu

        # Tahminler için single_helper fonksiyonunu çağır
        result = single_helper(train_images, sub_model, lelist)

        # Sonuçları JSON formatında döndür
        return jsonify({'predictions': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400



@app.route('/single_classification', methods=['POST'])
def single_classification_api():
    try:
        # Kullanıcıdan gelen resim dosyasını al
        img_file = request.files['image']

        # Dosyanın uzantısını kontrol et
        file_extension = os.path.splitext(img_file.filename)[1].lower()

        # Geçerli uzantılar: jpg, jpeg, png, vb.
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            return jsonify({'error': 'Invalid file extension'}), 400

        # Resmi geçici bir dosyaya kaydetmek yerine, belleğe alıyoruz
        img = Image.open(io.BytesIO(img_file.read()))  # BytesIO ile resmi alıyoruz
        
        # Resmi işle
        img.save('received_image.jpg')  # Geçici dosya kaydetme işlemi 

        # single_classification fonksiyonunu çağırarak sınıflandırma yap
        result_type, result_str, result_details = single_classification('received_image.jpg')

        # Sonuçları döndür
        return jsonify({
            'type': result_type,
            'result_str': result_str,
            'result_details': result_details
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)