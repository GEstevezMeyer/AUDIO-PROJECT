import pandas as pd 
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from multiprocessing import Pool, cpu_count, Queue , Process
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
    
    x, y, split = zip(*res)
    n_label = len(list(set(y))) 

    x = np.array(x)
    y = np.array(y)
    split = np.array(split)

    queue = Queue()

    px = Process(target=clean_x,args=(x,queue))
    py = Process(target= clean_y, args=(y,queue))

    px.start()
    py.start()
   
    x,y = organize_queue_outputs(queue)

    

    input_shape = x.shape[1::]

    train_mask = split == "TRAINING"
    test_mask  = split == "TEST"

    x_train = x[train_mask]
    y_train = y[train_mask]

    
    print(x_train.shape)
    print(y_train.shape)

    x_test = x[test_mask]
    y_test = y[test_mask]

    

    train_ds = tf.data.Dataset.from_tensor_slices((list(x_train), list(y_train)))
    test_ds  = tf.data.Dataset.from_tensor_slices((list(x_test), list(y_test)))

    train_ds = train_ds.shuffle(buffer_size=1000).batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)

    

    os.chdir("../../")
    return train_ds,test_ds,n_label,input_shape


def load_metadata() -> pd.DataFrame: 
    instrument_name =  os.path.basename(os.getcwd())
    df = pd.read_csv(f"metadata_{instrument_name}.csv")
    
    return df 




def clean_y(y:np.array,queue:Queue)-> np.array:
    unique_label = sorted(list(set(y))) 
    indices_label = {unique_label[i]:i for i in range(len(unique_label))}
    y_int = np.array([indices_label[x] for x in y], dtype=np.int32)
    n_classes = len(unique_label)
    y = one_hot_numba(y_int,n_classes)
    
    queue.put(("y",y))

def clean_x(x:np.array,queue:Queue):
    x_min = x.min()
    x_max = x.max()

    x = (x-x_min)/(x_max-x_min)

    x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)

    queue.put(("x",x))

def organize_queue_outputs(queue: Queue) -> tuple: 
    finish = False
    number_of_item_found = 0
    x = None
    y = None

    while not finish: 
        item = queue.get()
        if item is not None: 
            if item[0] == "x": 
                x = item[1]
            else: 
                y = item[1]
            number_of_item_found+= 1
        
        if number_of_item_found == 2: 
            finish = True

    return x,y 

@njit(parallel=True )
def one_hot_numba(y_int, n_classes):
    m = len(y_int)
    res = np.zeros((m, n_classes), dtype=np.int32)
    
    for i in range(m):
        res[i, y_int[i]] = 1
        
    return res


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
    print(main_process("DATA/GUITAR"))
    

