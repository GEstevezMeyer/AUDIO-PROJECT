import pandas as pd 
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from multiprocessing import Pool, cpu_count
import librosa
import time 
import numpy as np 
import tomllib
from functools import partial
import tensorflow as tf 
from numba import njit, prange



def import_config(path:str) -> dict: 
    with open(path, "rb") as r:
        config = tomllib.load(r)["config"]

    return config 


def main_process(directory_intrument:str,config_path:str = "functions/config.toml"):

    config = import_config(config_path)
    

    os.chdir(directory_intrument)
    metadata = load_metadata()
    func = partial(load_wavefile, config=config)

    
    with Pool(cpu_count()) as p:
        res = p.map(func, metadata.to_dict("records"))
    
    x1,x2, y, split = zip(*res)
    n_label = len(list(set(y))) 

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)
    split = np.array(split)

    if config["divide_4"] == True: 
        x1,y,split = divide_data(x1,y,split)

    input_shape1 = (x1.shape[1],x1.shape[2],1)
    input_shape2 = (x2.shape[1],x2.shape[2],1)

    train_mask = split == "TRAINING"
    test_mask  = split == "TEST"
    valTrue_mask = split == "VAL_REAL"

    x1_train = x1[train_mask]
    x1_test = x1[test_mask]
    x1_valTrue = x1[valTrue_mask]

    x2_train = x2[train_mask]
    x2_test = x2[test_mask]
    x2_valTrue = x2[valTrue_mask]

    x1 = (x1_train,x1_test,x1_valTrue)
    x2 = (x2_train,x2_test,x2_valTrue)

    with Pool(3) as p:
        res = p.map(clean_x, x1)
    
    x1_train,x1_test,x1_valTrue = res

    with Pool(3) as p:
        res = p.map(clean_x, x2)
    
    x2_train,x2_test,x2_valTrue = res

    y_train = y[train_mask]
    y_test = y[test_mask]
    y_valTrue = y[valTrue_mask]

    y = (y_test,y_train,y_valTrue)
    
    with Pool(3) as p:
        res = p.map(clean_y, y)

    y_test,y_train,y_valTrue = res

    print("training")
    print(y_train.shape,x1_train.shape)

    valTesting_df = (x1_valTrue,y_valTrue)

    valTesting_double_df = ((x1_valTrue,x2_valTrue),y_valTrue)
     
    train_ds = tf.data.Dataset.from_tensor_slices((list(x1_train), list(y_train)))
    test_ds  = tf.data.Dataset.from_tensor_slices((list(x1_test), list(y_test)))

    train_ds = train_ds.shuffle(buffer_size=1000).batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)


    train_double_ds = tf.data.Dataset.from_tensor_slices(
        ((x1_train, x2_train), y_train)
    )

    test_double_ds = tf.data.Dataset.from_tensor_slices(
        ((x1_test, x2_test), y_test)
    )

    train_double_ds = train_double_ds.shuffle(1000)\
        .batch(config["batch_size"])\
        .prefetch(tf.data.AUTOTUNE)

    test_double_ds = test_double_ds.batch(config["batch_size"])\
        .prefetch(tf.data.AUTOTUNE)
        
    print(input_shape1)
    print(input_shape2)

    os.chdir("../../")

    if config["model"] == "double":
        return train_double_ds,test_double_ds,n_label,input_shape1,input_shape2,valTesting_double_df
    else:
        return train_ds,test_ds,n_label,input_shape1,valTesting_df


def load_metadata() -> pd.DataFrame: 
    instrument_name =  os.path.basename(os.getcwd())
    df = pd.read_csv(f"metadata_{instrument_name}.csv")
    
    return df 



def divide_data(x:np.array,y:np.array,split:np.array) -> tuple: 
    
    new_x = []
    new_y = []
    new_split = []

    for xi, yi, si in zip(x, y, split):
        parts = np.array_split(xi, 4)  
        
        new_x.extend(parts)
        new_y.extend([yi]*4)
        new_split.extend([si]*4)

    x = np.array(new_x)
    y = np.array(new_y)
    split = np.array(new_split)

    return x,y,split

def clean_y(y:np.array)-> np.array:
    unique_label = sorted(list(set(y))) 
    indices_label = {unique_label[i]:i for i in range(len(unique_label))}
    y_int = np.array([indices_label[x] for x in y], dtype=np.int32)
    n_classes = len(unique_label)
    y = one_hot_numba(y_int,n_classes)
    
    return y 

def clean_x(x:np.array):
    x_min = x.min()
    x_max = x.max()

    x = (x-x_min)/(x_max-x_min)

    x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)

    return x


@njit(parallel=True)
def one_hot_numba(y_int, n_classes):
    m = len(y_int)
    res = np.zeros((m, n_classes), dtype=np.int32)
    
    for i in range(m):
        res[i, y_int[i]] = 1
        
    return res


def load_wavefile(series:dict,config:dict): 

    path_file = create_path(series)

    waveform, sample_rate = librosa.load(path=path_file, sr=16000)

    waveform,_ = librosa.effects.hpss(waveform)

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

    mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=config["nmels"]
    )

    x1 = mel_spectrogram
    x2 = mfcc
    y = series["label"]
    split = series["Split"]



    
    return x1,x2,y,split



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
    print(main_process("DATA/GUITAR"))
    

