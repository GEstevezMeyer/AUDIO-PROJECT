import os 
import pandas as pd
from multiprocessing import Pool
from pathlib import Path


def main(directory_data: str) -> int: 
    os.chdir(directory_data)
    instruments = os.listdir()
    for instrument in instruments: 
        make_instrument_metadata(instrument)
    return 0

def make_instrument_metadata(directory_instrument: str): 
    os.chdir(directory_instrument)
    elements = os.listdir()

    directory_instrument = Path(directory_instrument)
    sets = []

    for element in elements: 
        if element != f"metadata_{directory_instrument}.csv": 
            sets.append(element)
    
    with Pool(2) as p:
        dfs = p.map(make_metadata, sets)

    df = pd.concat([dfs[0], dfs[1]])
    df.to_csv(f"metadata_{directory_instrument}.csv",index= False)
    os.chdir("../")

def make_metadata(directory_data: str): 
    labels = os.listdir(directory_data)
    df = pd.DataFrame(columns=["id_audio", "label", "Split"])
    for label in labels: 
        audios = os.listdir(directory_data + "/" + label)
        for audio in audios: 
            df.loc[len(df)] = [audio, label, directory_data]

    return df

if __name__ == "__main__":
    print(main("DATA"))
    