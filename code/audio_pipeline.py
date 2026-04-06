import pandas as pd 
import os 
from multiprocessing import Pool, cpu_count
import librosa
import time 
import numpy as np 
import tomllib
from functools import partial
import tensorflow as tf 



def import_config(path:str) -> dict: 
    with open(path, "rb") as r:
        config = tomllib.load(r)["config"]

    return config 


def main_process(directory_intrument:str):
    config = import_config("code/config.toml")

    os.chdir(directory_intrument)
    metadata = load_metadata()
    func = partial(load_wavefile, config=config)

    
    with Pool(cpu_count()) as p: 
        res = p.map(func,metadata.to_dict("records"))
    
    x, y, split = zip(*res)

    x = np.array(x)
    y = np.array(y)
    split = np.array(split)
   
    train_mask = split == "TRAINING"
    test_mask  = split == "TEST"

    x_train = x[train_mask]
    y_train = y[train_mask]

    x_test = x[test_mask]
    y_test = y[test_mask]

    train_ds = tf.data.Dataset.from_tensor_slices((list(x_train), list(y_train)))
    test_ds  = tf.data.Dataset.from_tensor_slices((list(x_test), list(y_test)))

    train_ds = train_ds.shuffle(buffer_size=1000).batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)



    os.chdir("../../")
    return x_train,y_train,x_test,y_test,


def load_metadata() -> pd.DataFrame: 
    instrument_name =  os.path.basename(os.getcwd())
    df = pd.read_csv(f"metadata_{instrument_name}.csv")
    
    return df 

def load_wavefile(series:dict,config:dict): 

    path_file = create_path(series)

    waveform, sample_rate = librosa.load(path=path_file, sr=16000)

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
    y = series["label"]
    split = series["Split"]



    
    return x,y,split 



def create_path(row:pd.Series) -> str:
    return f"{row['Split']}/{row['label']}/{row['id_audio']}"

def enveloppe(waveform,sample_rate,threshold = 0.0005): 
    mask = []
    y = pd.Series(waveform).map(lambda x: np.abs(x))
    y_mean = y.rolling(window=int(sample_rate/10),min_periods=1,center=True).mean()

    for mean in y_mean: 
        if mean > threshold: 
            mask.append(True)
        else: 
            mask.append(False)

    return mask

def pad_waveform(waveform, target_length):
    if len(waveform) < target_length:
        pad_width = target_length - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode='constant')
    else:
        waveform = waveform[:target_length]
    return waveform





if __name__ == "__main__": 
    x,y,_,_ = main_process("DATA/GUITAR")

    print(x.shape,y.shape)

    

