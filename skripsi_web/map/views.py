# Create your views here.
from django.shortcuts import render
from .models import TrainingData
from .models import MyModel
import folium
from django.http import HttpResponse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import keras
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix   

from tensorflow.keras.models import load_model

import base64
from io import BytesIO

from django.shortcuts import render

from PIL import Image
from io import BytesIO
import base64

import json


from sklearn.metrics import confusion_matrix

def get_default_icon_url(color):
    default_icon_url = f"https://raw.githubusercontent.com/python-visualization/folium/master/folium/templates/icons/{color}.png"
    return default_icon_url


def create_color_square(color, size=20):
    image = Image.new("RGB", (size, size), color)
    image_data = BytesIO()
    image.save(image_data, format="PNG")
    encoded_image = base64.b64encode(image_data.getvalue())
    return encoded_image

def predict(request):
 

    if request.method == 'POST':
        # Get the form data
        
        sentralitas = float(request.POST.get('sentralitas'))
        visibilitas = float(request.POST.get('visibilitas'))
        bangunan = float(request.POST.get('bangunan'))
        luas = float(request.POST.get('luas'))

        # Perform database operations with the form data
        # Example: Saving the data to a Django model
        # from .models import MyModel
        # my_model = MyModel(peruntukan=peruntukan, sentralitas=sentralitas, visibilitas=visibilitas, bangunan=bangunan, luas=luas)
        # my_model.save()

        # Return a JSON response indicating success
        input_data = {
        "Jarak_pusat_kota2": np.array([sentralitas]),
        "Visibilitas": np.array([visibilitas]),
        "Bangunan": np.array([bangunan]),
        "Luas": np.array([luas])
    }

        # Create a list to store the input tensors
        input_tensors = []

        # Iterate over the input data dictionary and convert each feature to a tensor
        for feature_name, feature_data in input_data.items():
            feature_tensor = tf.convert_to_tensor(feature_data)
            input_tensors.append(feature_tensor)

        
        
        column_list = ["Jarak_pusat_kota2", "Visibilitas", "Bangunan", "Luas"]

        # Create DataFrame
        #df = pd.DataFrame([input_data], columns=column_list)
        #val = df_to_dataset_val(df)

        model = load_model("data_3/50.h5")
        

        # Lakukan prediksi menggunakan model
        y_pred = model.predict(input_tensors)
        label_kelas = ['Sawah', 'Perumahan', 'Taman', 'Ruko', 'Kantor', 'Pasar']
        y_pred_label = [label_kelas[np.argmax(prediksi)] for prediksi in y_pred]

    
        peruntukan = y_pred_label
        hasil = y_pred
        my_model = MyModel(peruntukan = peruntukan, sentralitas=sentralitas, visibilitas=visibilitas, bangunan=bangunan, luas=luas, hasil=json.dumps(hasil.tolist()))
        my_model.save()

        

        
    else:
        print("else")
    


    data = MyModel.objects.all()

    # Pass the data to the template
    context = {'data': data}

    return render(request, 'predict.html', context)


def asset_map(request):
  # membuat peta folium
    madiun_map = folium.Map(location=[-7.6310195, 111.5238517], zoom_start=13,
    width="100%",
    height="100%",
    control_scale=True)

    color_mapping = {
    'Sawah': 'orange',
    'Perumahan': 'purple',
    'Taman': 'blue',
    'Ruko': 'green',
    'Kantor': 'yellow',
    'Pasar': 'red',
    # Add more categories and colors as needed
    }

        


    # menambahkan marker pada peta folium untuk setiap data TrainingData
    for data in TrainingData.objects.all():
        
        peruntukan = data.Peruntukan

        if peruntukan in color_mapping:
            icon_color = color_mapping[peruntukan]
        else:
            icon_color = 'red'  # Default color if category not found in mapping
        
        popup_html = f"<b>Peruntukan:</b> {data.Peruntukan}<br><b>Pusat Kota:</b> {data.Pusat_kota}<br><b>Visibilitas:</b> {data.Visibilitas}<br><b>Bangunan:</b> {data.Bangunan}<br><b>Luas:</b> {data.Luas} m<sup>2</sup><br><b>Latitude:</b> {data.latitude}<br><b>Longitude:</b> {data.longitude}<br><b>Deskripsi:</b> {data.Deskripsi}"
        folium.Marker(
            location=[data.latitude * -1, data.longitude],
            popup=popup_html,
            icon=folium.Icon(color=icon_color)
        ).add_to(madiun_map)

    #menambahkan atribut data kedalam context
    tabel_data = TrainingData.objects.all()



    csv = pd.read_csv("data_3/rumus + data skripsi angka polos jarak dan visibilitas 2.csv")
    csv_data = csv.to_dict(orient='records')

    #prediksi, asli =perform_predict('data_3/10.h5', 0.1)



    legend_html = ''
    for category, color in color_mapping.items():
        color_square = create_color_square(color)
        legend_html += f"<div><img src='data:image/png;base64,{color_square.decode('utf-8')}' class='legend-icon'>{category}</div>"


    

    # rendering peta dan tabel ke template
    context = {'madiun_map': madiun_map._repr_html_(),
                'data': csv_data,
                'legend_html': legend_html,
                }

    
    


    return render(request, 'index.html', context)


def sistem3_view(request):
    # membuat peta folium
    madiun_map = folium.Map(location=[-7.6310195, 111.5238517], zoom_start=13,
    width="100%",
    height="100%",
    control_scale=True)

    # menambahkan marker pada peta folium untuk setiap data TrainingData
    for data in TrainingData.objects.all():
        popup_html = f"<b>Peruntukan:</b> {data.Peruntukan}<br><b>Pusat Kota:</b> {data.Pusat_kota}<br><b>Visibilitas:</b> {data.Visibilitas}<br><b>Bangunan:</b> {data.Bangunan}<br><b>Luas:</b> {data.Luas} m<sup>2</sup><br><b>Latitude:</b> {data.latitude}<br><b>Longitude:</b> {data.longitude}<br><b>Deskripsi:</b> {data.Deskripsi}"
        folium.Marker(
            location=[data.latitude * -1, data.longitude],
            popup=popup_html,
            icon=folium.Icon(color='red')
        ).add_to(madiun_map)

    #menambahkan atribut data kedalam context
         



    #data = pd.read_csv("data_3/rumus + data skripsi angka polos jarak dan visibilitas 2.csv")
    prediksi, asli, cm ,akurasi, presisi, recall, t_akurasi, t_presisi, t_recall, t_f1 =perform_predict('data_3/30.h5', 0.3)


    true_positives = np.diagonal(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    labels=["Sawah", "Perumahan", "Taman", "Ruko","Kantor","Pasar"]


    tp = zip(labels, true_positives, false_positives, false_negatives)
    
    data = TrainingData.objects.all()
    

    # rendering peta dan tabel ke template
    context = {'madiun_map': madiun_map._repr_html_(),
                'data ': data,
                'prediksi': prediksi,
                'asli': asli,
                'confusion_matrix': cm,
                'akurasi': akurasi,
                'presisi': presisi,
                'recall': recall,
                't_akurasi': t_akurasi,
                't_presisi': t_presisi,
                't_recall': t_recall,
                'tp':tp,
                't_f1' : t_f1,
                }



    return render(request, 'index3.html', context)

def sistem1_view(request):
    # membuat peta folium
    madiun_map = folium.Map(location=[-7.6310195, 111.5238517], zoom_start=13,
    width="100%",
    height="100%",
    control_scale=True)

    # menambahkan marker pada peta folium untuk setiap data TrainingData
    for data in TrainingData.objects.all():
        popup_html = f"<b>Peruntukan:</b> {data.Peruntukan}<br><b>Pusat Kota:</b> {data.Pusat_kota}<br><b>Visibilitas:</b> {data.Visibilitas}<br><b>Bangunan:</b> {data.Bangunan}<br><b>Luas:</b> {data.Luas} m<sup>2</sup><br><b>Latitude:</b> {data.latitude}<br><b>Longitude:</b> {data.longitude}<br><b>Deskripsi:</b> {data.Deskripsi}"
        folium.Marker(
            location=[data.latitude * -1, data.longitude],
            popup=popup_html,
            icon=folium.Icon(color='red')
        ).add_to(madiun_map)

    #menambahkan atribut data kedalam context
         
    data = TrainingData.objects.all()


    #data = pd.read_csv("data_3/rumus + data skripsi angka polos jarak dan visibilitas 2.csv")
    prediksi, asli, cm ,akurasi, presisi, recall, t_akurasi, t_presisi, t_recall, t_f1 =perform_predict('data_3/10.h5', 0.1)


    true_positives = np.diagonal(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    labels=["Sawah", "Perumahan", "Taman", "Ruko","Kantor","Pasar"]


    tp = zip(labels, true_positives, false_positives, false_negatives)
    

    

    # rendering peta dan tabel ke template
    context = {'madiun_map': madiun_map._repr_html_(),
                'data ': data,
                'prediksi': prediksi,
                'asli': asli,
                'confusion_matrix': cm,
                'akurasi': akurasi,
                'presisi': presisi,
                'recall': recall,
                't_akurasi': t_akurasi,
                't_presisi': t_presisi,
                't_recall': t_recall,
                'tp':tp,
                't_f1' : t_f1,
                }



    return render(request, 'index3.html', context)

def sistem2_view(request):
    # membuat peta folium
    madiun_map = folium.Map(location=[-7.6310195, 111.5238517], zoom_start=13,
    width="100%",
    height="100%",
    control_scale=True)

    # menambahkan marker pada peta folium untuk setiap data TrainingData
    for data in TrainingData.objects.all():
        popup_html = f"<b>Peruntukan:</b> {data.Peruntukan}<br><b>Pusat Kota:</b> {data.Pusat_kota}<br><b>Visibilitas:</b> {data.Visibilitas}<br><b>Bangunan:</b> {data.Bangunan}<br><b>Luas:</b> {data.Luas} m<sup>2</sup><br><b>Latitude:</b> {data.latitude}<br><b>Longitude:</b> {data.longitude}<br><b>Deskripsi:</b> {data.Deskripsi}"
        folium.Marker(
            location=[data.latitude * -1, data.longitude],
            popup=popup_html,
            icon=folium.Icon(color='red')
        ).add_to(madiun_map)

    #menambahkan atribut data kedalam context
         



    #data = pd.read_csv("data_3/rumus + data skripsi angka polos jarak dan visibilitas 2.csv")
    prediksi, asli, cm ,akurasi, presisi, recall, t_akurasi, t_presisi, t_recall, t_f1 =perform_predict('data_3/20.h5', 0.2)


    true_positives = np.diagonal(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    labels=["Sawah", "Perumahan", "Taman", "Ruko","Kantor","Pasar"]


    tp = zip(labels, true_positives, false_positives, false_negatives)
    

    

    # rendering peta dan tabel ke template
    context = {'madiun_map': madiun_map._repr_html_(),
                'data ': data,
                'prediksi': prediksi,
                'asli': asli,
                'confusion_matrix': cm,
                'akurasi': akurasi,
                'presisi': presisi,
                'recall': recall,
                't_akurasi': t_akurasi,
                't_presisi': t_presisi,
                't_recall': t_recall,
                'tp':tp,
                't_f1' : t_f1,
                }


    return render(request, 'index3.html', context)


    # membuat peta folium
    madiun_map = folium.Map(location=[-7.6310195, 111.5238517], zoom_start=13,
    width="100%",
    height="100%",
    control_scale=True)

    # menambahkan marker pada peta folium untuk setiap data TrainingData
    for data in TrainingData.objects.all():
        popup_html = f"<b>Peruntukan:</b> {data.Peruntukan}<br><b>Pusat Kota:</b> {data.Pusat_kota}<br><b>Visibilitas:</b> {data.Visibilitas}<br><b>Bangunan:</b> {data.Bangunan}<br><b>Luas:</b> {data.Luas} m<sup>2</sup><br><b>Latitude:</b> {data.latitude}<br><b>Longitude:</b> {data.longitude}<br><b>Deskripsi:</b> {data.Deskripsi}"
        folium.Marker(
            location=[data.latitude * -1, data.longitude],
            popup=popup_html,
            icon=folium.Icon(color='red')
        ).add_to(madiun_map)

    #menambahkan atribut data kedalam context
    data = TrainingData.objects.all()



    #data = pd.read_csv("data_3/rumus + data skripsi angka polos jarak dan visibilitas 2.csv")
    prediksi, asli =perform_predict('data_3/20.h5', 0.2)

    

    

    # rendering peta dan tabel ke template
    context = {'madiun_map': madiun_map._repr_html_(),
                'data ': data,
                'prediksi': prediksi,
                'asli': asli,

                }


    return render(request, 'index3.html', context)

def sistem4_view(request):
    # membuat peta folium
    madiun_map = folium.Map(location=[-7.6310195, 111.5238517], zoom_start=13,
    width="100%",
    height="100%",
    control_scale=True)

    # menambahkan marker pada peta folium untuk setiap data TrainingData
    for data in TrainingData.objects.all():
        popup_html = f"<b>Peruntukan:</b> {data.Peruntukan}<br><b>Pusat Kota:</b> {data.Pusat_kota}<br><b>Visibilitas:</b> {data.Visibilitas}<br><b>Bangunan:</b> {data.Bangunan}<br><b>Luas:</b> {data.Luas} m<sup>2</sup><br><b>Latitude:</b> {data.latitude}<br><b>Longitude:</b> {data.longitude}<br><b>Deskripsi:</b> {data.Deskripsi}"
        folium.Marker(
            location=[data.latitude * -1, data.longitude],
            popup=popup_html,
            icon=folium.Icon(color='red')
        ).add_to(madiun_map)

    #menambahkan atribut data kedalam context
         



    #data = pd.read_csv("data_3/rumus + data skripsi angka polos jarak dan visibilitas 2.csv")
    prediksi, asli, cm ,akurasi, presisi, recall, t_akurasi, t_presisi, t_recall, t_f1 =perform_predict('data_3/40.h5', 0.4)


    true_positives = np.diagonal(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    labels=["Sawah", "Perumahan", "Taman", "Ruko","Kantor","Pasar"]


    tp = zip(labels, true_positives, false_positives, false_negatives)
    

    #menambahkan atribut data kedalam context
    data = TrainingData.objects.all()


    # rendering peta dan tabel ke template
    context = {'madiun_map': madiun_map._repr_html_(),
                'data ': data,
                'prediksi': prediksi,
                'asli': asli,
                'confusion_matrix': cm,
                'akurasi': akurasi,
                'presisi': presisi,
                'recall': recall,
                't_akurasi': t_akurasi,
                't_presisi': t_presisi,
                't_recall': t_recall,
                'tp':tp,
                't_f1' : t_f1,
                }


    return render(request, 'index3.html', context)



    #menambahkan atribut data kedalam context
    data = TrainingData.objects.all()




def sistem5_view(request):
    # membuat peta folium
    madiun_map = folium.Map(location=[-7.6310195, 111.5238517], zoom_start=13,
    width="100%",
    height="100%",
    control_scale=True)

    # menambahkan marker pada peta folium untuk setiap data TrainingData
    for data in TrainingData.objects.all():
        popup_html = f"<b>Peruntukan:</b> {data.Peruntukan}<br><b>Pusat Kota:</b> {data.Pusat_kota}<br><b>Visibilitas:</b> {data.Visibilitas}<br><b>Bangunan:</b> {data.Bangunan}<br><b>Luas:</b> {data.Luas} m<sup>2</sup><br><b>Latitude:</b> {data.latitude}<br><b>Longitude:</b> {data.longitude}<br><b>Deskripsi:</b> {data.Deskripsi}"
        folium.Marker(
            location=[data.latitude * -1, data.longitude],
            popup=popup_html,
            icon=folium.Icon(color='red')
        ).add_to(madiun_map)

    #menambahkan atribut data kedalam context
         



    #data = pd.read_csv("data_3/rumus + data skripsi angka polos jarak dan visibilitas 2.csv")
    prediksi, asli, cm ,akurasi, presisi, recall, t_akurasi, t_presisi, t_recall, t_f1 =perform_predict('data_3/50.h5', 0.5)


    true_positives = np.diagonal(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives
    labels=["Sawah", "Perumahan", "Taman", "Ruko","Kantor","Pasar"]


    tp = zip(labels, true_positives, false_positives, false_negatives)
    


    #menambahkan atribut data kedalam context
    data = TrainingData.objects.all()
    

    # rendering peta dan tabel ke template
    context = {'madiun_map': madiun_map._repr_html_(),
                'data ': data,
                'prediksi': prediksi,
                'asli': asli,
                'confusion_matrix': cm,
                'akurasi': akurasi,
                'presisi': presisi,
                'recall': recall,
                't_akurasi': t_akurasi,
                't_presisi': t_presisi,
                't_recall': t_recall,
                'tp':tp,
                't_f1' : t_f1,
                }


    return render(request, 'index3.html', context)


   




def get_data_with_ratio(ratio,data):
    # Shuffle the data
    shuffled_data = data

    # Get unique classes from the "Peruntukan" column
    classes = shuffled_data['Peruntukan'].unique()

    # Initialize empty dataframes for train and test data
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # Iterate over each class
    for cls in classes:
        cls_data = shuffled_data[shuffled_data['Peruntukan'] == cls]

        # Split the class data into train and test
        cls_train, cls_test = train_test_split(cls_data, test_size=ratio, random_state = 102,shuffle = True)

        # Append the train and test data to the respective dataframes
        train_data = train_data.append(cls_train)
        test_data = test_data.append(cls_test)
        
    training = train_data
    test = test_data

    training = training[["Peruntukan","Jarak_pusat_kota2","Visibilitas","Bangunan","Luas"]]
    test = test[["Peruntukan","Jarak_pusat_kota2","Visibilitas","Bangunan","Luas"]]


    return training, test



def change_categorical_to_number(data):
    
    data = data
    
 
    ####
    ####

    condition = [  data.Visibilitas == "Strategis",
                 data.Visibilitas == "Sedang",
                  data.Visibilitas == "Kurang",
    ]

    value = [3,2,1]

    data.Visibilitas = np.select(condition,value)
    #####
    #####

    condition = [  data.Bangunan == "Bagus",
                 data.Bangunan == "Sedang",

    ]

    value = [2,1]

    data.Bangunan = np.select(condition,value,0)


    condition = [ 
                data.Peruntukan == "Pasar",
                 data.Peruntukan == "Kantor",
                 data.Peruntukan == "Ruko",
                 data.Peruntukan == "Taman",
                 data.Peruntukan == "Perumahan",
                 data.Peruntukan == "Sawah",

    ]

    value = [5,4,3,2,1,0]

    data.Peruntukan = np.select(condition,value,0)
    
    return data

def df_to_dataset(dataframe, shuffle=False, batch_size=1):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Peruntukan')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def df_to_dataset_val(dataframe, shuffle=False, batch_size=1):
  dataframe = dataframe.copy()
  #labels = dataframe.pop('Peruntukan')
  labels = ["Jarak_pusat_kota2", "Visibilitas", "Bangunan", "Luas"]

  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

def encoded_features(train_ds):
    all_inputs = []
    encoded_features = []

    # Numerical features.
    for header in ["Jarak_pusat_kota2","Visibilitas","Bangunan","Luas"]:
      numeric_col = tf.keras.Input(shape=(1,), name=header)
      normalization_layer = get_normalization_layer(header, train_ds)
      encoded_numeric_col = normalization_layer(numeric_col)
      all_inputs.append(numeric_col)
      encoded_features.append(encoded_numeric_col)
    return encoded_features, all_inputs



def angka_to_label(data):

    condition = [ 
                data.Peruntukan == "Pasar",
                 data.Peruntukan == "Kantor",
                 data.Peruntukan == "Ruko",
                 data.Peruntukan == "Taman",
                 data.Peruntukan == "Perumahan",
                 data.Peruntukan == "Sawah",

    ]

    value = [5,4,3,2,1,0]

    np.select


def perform_predict(model_name, ratio):
    # Memuat model dari file .h5
    model = load_model(model_name)

    data = pd.read_csv("data_3/rumus + data skripsi angka polos jarak dan visibilitas 2.csv")

    train, test2 = get_data_with_ratio(ratio, data)

    data_asli = test2.copy()  # Create a copy of test2 DataFrame
    data_prediksi = test2.copy()  # Create a separate copy for predicted results

    test = change_categorical_to_number(test2)

    testing = df_to_dataset(test)
    encoded_testing, all_inputs = encoded_features(testing)

    # Lakukan prediksi menggunakan model
    y_pred = model.predict(testing)
    label_kelas = ['Sawah', 'Perumahan', 'Taman', 'Ruko', 'Kantor', 'Pasar']
    y_pred_label = [label_kelas[np.argmax(prediksi)] for prediksi in y_pred]

    prediksi = y_pred_label

    

    data_prediksi.Peruntukan = y_pred_label

    df_prediksi = data_prediksi.to_dict(orient='records')
    df_asli = data_asli.to_dict(orient='records')

    y_true = data_prediksi.Peruntukan
    y_pred = data_asli.Peruntukan

    cm ,akurasi, presisi, recall, t_akurasi, t_presisi, t_recall, t_f1 = confussion_matrix(y_true, y_pred)
    
    

    return df_prediksi, df_asli, cm ,akurasi, presisi, recall, t_akurasi, t_presisi, t_recall, t_f1


def confussion_matrix(test_labels, y_pred_label):
     
    test_labels = list(test_labels)
    y_pred_label = list(y_pred_label)

    
    cm = confusion_matrix(test_labels, y_pred_label,labels=["Sawah", "Perumahan", "Taman", "Ruko","Kantor","Pasar"])
    print("aktual test")
    print(test_labels)
    print("print print print prediksi test")
    print(y_pred_label)

    print('Confusion matrix:')
    print(cm)
    

    
    true_positives = np.diagonal(cm)

    # Step 3: Calculate the accuracy percentage for each class
    class_data_totals = np.sum(cm, axis=0)
    class_accuracies = true_positives / class_data_totals * 100

    # Step 4: Calculate the overall accuracy percentage
    total_instances = np.sum(cm)
    overall_accuracy = np.sum(true_positives) / total_instances * 100
    
    
    true_positives = np.diagonal(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives




    # Step 3: Calculate precision, recall, and F1 score for each label
    accuracy = true_positives / np.sum(cm, axis=1) * 100
    precision = true_positives / (true_positives + false_positives) * 100
    recall = true_positives / (true_positives + false_negatives) * 100
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Step 4: Calculate the total accuracy
    mean_accuracy = np.mean(accuracy)

    total_accuracy = overall_accuracy
    total_precision = np.mean(precision)
    total_recall = np.mean(recall)
    total_f1 = np.mean(f1_score)

    # Print the results
    print("Total Accuracy:", total_accuracy)
    print("Total Precision:", total_precision)
    print("Total Recall:", total_recall)
    
    print("TP for each label", true_positives)
    print("Total data for each label",  np.sum(cm, axis=1))
    print("FP for each label", np.sum(cm, axis=0) - true_positives)
    print("FN for each label", np.sum(cm, axis=1) - true_positives)
    # Print the results
    print("Accuracy for each label:", class_accuracies)
    print("Precision for each label:", precision)
    print("Recall for each label:", recall)

    return cm ,precision, recall, f1_score ,total_accuracy,total_precision,total_recall,total_f1

    
