import tensorflow as tf 
from audio_pipeline import main_process,enveloppe,import_config,pad_waveform
import librosa
import matplotlib as plt 
import tomllib 
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np 



def main_training(data_path:str,config_path:str = "code/config.toml"): 

    parameters = import_parameters(config_path)
    train_dataset,val_dataset,n_label,input_shape = main_process(data_path)
    model = create_conv_model(input_shape,n_label)

    metrics = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=parameters["epochs"],shuffle= True)
    
    return model,metrics 
    



def create_conv_model(input_shape:tuple,number_of_label:int)-> tf.keras.Sequential: 
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = input_shape),
        tf.keras.layers.Conv2D(16,(3,3),activation="relu",strides=(1,1),padding="same"),
        tf.keras.layers.Conv2D(32,(3,3),activation="relu",strides=(1,1),padding="same"),
        tf.keras.layers.Conv2D(64,(3,3),activation="relu",strides=(1,1),padding="same"),
        tf.keras.layers.Conv2D(128,(3,3),activation="relu",strides=(1,1),padding="same"),
        
        tf.keras.layers.MaxPool2D((2,2)),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(32,activation="relu"),
        tf.keras.layers.Dense(number_of_label,activation="softmax")
    ])
    model.compile(loss= "categorical_crossentropy",optimizer= "adam",metrics= ["acc"])
    model.summary()

    return model 

def import_parameters(path:str) -> dict: 
    with open(path, "rb") as r:
        config = tomllib.load(r)["parameters"]
    return config


def predict_chord(model,wave_file_path:str,config_path = "code/config.toml"): 

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
    print(predict_chord(model,"DATA/GUITAR/TEST/Am/Am_Classic_Jo_4.wav"))
