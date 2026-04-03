import pandas as pd 
import os 
from multiprocessing import Pool, cpu_count
import librosa
import time 


def main_process(directory_intrument:str):
    os.chdir(directory_intrument)
    metadata = load_metadata()

    
    with Pool(cpu_count()) as p: 
        p.map(load_wavefile,metadata.to_dict("records"))

    os.chdir("../../")
    return 0

def main_sequential(directory_intrument: str):
    os.chdir(directory_intrument)
    metadata = load_metadata()

    results = []
    for row in metadata.to_dict("records"):
        res = load_wavefile(row)
        results.append(res)

    os.chdir("../../")
    return 0



def load_metadata() -> pd.DataFrame: 
    instrument_name =  os.path.basename(os.getcwd())
    df = pd.read_csv(f"metadata_{instrument_name}.csv")
    
    return df 

def load_wavefile(series:dict): 

    path_file = create_path(series)
    waveform, sample_rate = librosa.load(path= path_file)
    res = ((waveform,sample_rate),series["label"])

    return res

def create_path(row:pd.Series) -> str:
    return f"{row['Split']}/{row['label']}/{row['id_audio']}"


if __name__ == "__main__": 

    

    start = time.time()
    main_process("DATA/GUITAR")
    end = time.time()

    print(f"Time: {end - start:.4f} seconds")

    start = time.time()
    main_sequential("DATA/GUITAR")
    end = time.time()

    print(f"Time: {end - start:.4f} seconds")

