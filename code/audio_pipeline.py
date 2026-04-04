import pandas as pd 
import os 
from multiprocessing import Pool, cpu_count
import librosa
import time 
import numpy as np 
import scipy.io as wavfile 
from python_speech_features import mfcc, logfbank


def main_process(directory_intrument:str):
    os.chdir(directory_intrument)
    metadata = load_metadata()

    
    with Pool(cpu_count()) as p: 
        p.map(load_wavefile,metadata.to_dict("records"))

    os.chdir("../../")

    return p

def main_sequential(directory_intrument: str):
    os.chdir(directory_intrument)
    metadata = load_metadata()

    results = []
    for row in metadata.to_dict("records"):
        res = load_wavefile(row)
        results.append(res)

    os.chdir("../../")
    return res


def load_metadata() -> pd.DataFrame: 
    instrument_name =  os.path.basename(os.getcwd())
    df = pd.read_csv(f"metadata_{instrument_name}.csv")
    
    return df 

def load_wavefile(series:dict): 

    path_file = create_path(series)
    waveform, sample_rate = librosa.load(path= path_file,sr= 16000)
    mask = enveloppe(waveform,sample_rate)
    waveform = waveform[mask]

    magnitud,frequency = calc_fft(waveform,sample_rate)
    bank = logfbank(waveform[:sample_rate],sample_rate,nfilt=26,nfft=1024).T
    mel = mfcc(waveform[:sample_rate],sample_rate,numcep=13,nfilt=26,nfft=1024).T

    res = {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "magnitud":magnitud,
        "frequency": frequency,
        "filterbank": bank,
        "mel": mel,
        "label": series["label"]
    }

    res = ((waveform,sample_rate),series["label"])

    return res

def create_path(row:pd.Series) -> str:
    return f"{row['Split']}/{row['label']}/{row['id_audio']}"


def calc_fft(signal,rate): 
    n = len(signal)
    frequency = np.fft.rfftfreq(n,d= 1/rate)
    magnitud = np.abs(np.fft.rfft(signal)/n)

    return (magnitud,frequency)

def enveloppe(waveform,sample_rate,treshold = 0.0005): 
    mask = []
    y = pd.Series(waveform).map(lambda x: np.abs(x))
    y_mean = y.rolling(window=int(sample_rate/10),min_periods=1,center=True).mean()

    for mean in y_mean: 
        if mean > treshold: 
            mask.append(True)
        else: 
            mask.append(False)

    return mask


if __name__ == "__main__": 

    

    start = time.time()
    main_process("DATA/GUITAR")
    end = time.time()

    print(f"Time: {end - start:.4f} seconds")

    start = time.time()
    main_sequential("DATA/GUITAR")
    end = time.time()

    print(f"Time: {end - start:.4f} seconds")

