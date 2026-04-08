
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from audio_pipeline import main_process,enveloppe,import_config,pad_waveform

import tensorflow as tf
import librosa
import matplotlib as plt 
import tomllib 
import numpy as np 
import mlflow 


def main_training(data_path:str,config_path:str = "functions/config.toml",ml_flow_url:str = "http://localhost:5000",experiment_name = "AUDIO_CNN"):

    mlflow.set_tracking_uri(ml_flow_url) 
    mlflow.set_experiment(experiment_name)

    parameters = import_parameters(config_path)
    config = import_config(config_path)
    train_dataset,val_dataset,n_label,input_shape = main_process(data_path)
    model = create_conv_model(input_shape,n_label,parameters["number_of_conv_layers"],parameters["filter_start"],parameters["step_size"])

    with mlflow.start_run(): 

        mlflow.log_params(config)
        mlflow.log_params(parameters)

        metrics = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=parameters["epochs"],shuffle= True)
        
        mlflow.tensorflow.log_model(model,"audio_IA")

        for epoch in range(len(metrics.history["loss"])):
            mlflow.log_metrics(
                {k: float(v[epoch]) for k, v in metrics.history.items()},
                step=epoch
            )
        
    
    return model,metrics 
    



def create_conv_model(input_shape:tuple,number_of_label:int,number_of_conv_layer:int = 3,filter_start:int = 16,step_size:int = 16)-> tf.keras.Sequential: 
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape))

    for i in  range (number_of_conv_layer):
        filters = filter_start + i*step_size
        model.add(tf.keras.layers.Conv2D(filters,(3,3), activation="relu", strides=(1,1), padding="same"))

    model.add(tf.keras.layers.MaxPool2D((2,2)))

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(number_of_label, activation="softmax"))


    model.compile(loss= "categorical_crossentropy",optimizer= "adam",metrics= ["acc"])
    model.summary()

    return model 

def import_parameters(path:str) -> dict: 
    with open(path, "rb") as r:
        config = tomllib.load(r)["parameters"]
    return config


def predict_chord(model,wave_file_path:str,config_path = "functions/config.toml"): 

    config = import_config(config_path)

    waveform, sample_rate = librosa.load(path=wave_file_path, sr=16000)

    mask = enveloppe(waveform, sample_rate)
    waveform = waveform[mask]
    waveform = pad_waveform(waveform, config["target_length"])

    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=config["nfft"],
        hop_length=config["hop_length"],
        n_mels=config["nmels"]
    )

    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    x = mel_spectrogram

    x_min = x.min()
    x_max = x.max()

    x = (x-x_min)/(x_max-x_min)
    x = x.reshape(1,x.shape[0],x.shape[1],1)

    res = model.predict(x)
    
    return res 

  
if __name__ == "__main__":  
    model,_ = main_training("DATA/GUITAR")
    
