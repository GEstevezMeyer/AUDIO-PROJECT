import numpy as np 
import matplotlib.pyplot as plt 
import pygame
from multiprocessing import Process,Queue
from functools import partial
import pyaudio
from training_model import import_config
import librosa
import mlflow.pyfunc
from audio_pipeline import enveloppe,pad_waveform

mlflow.set_tracking_uri("http://localhost:5000")


def create_random_point(radius:float,noise_bias:float = 0.05) -> tuple: 
    z = np.random.uniform(-radius,radius)
    theta = np.random.uniform(0,2*np.pi)
    x = np.sqrt(radius**2-z**2)*np.cos(theta)+np.random.normal(0,noise_bias)
    y = np.sqrt(radius**2-z**2)*np.sin(theta)+np.random.normal(0,noise_bias)

    return (x,y,z)


def project(point, width, height, scale=200, d=3):
    x, y, z = point
    factor = scale / (z + d)
    x2d = int(x * factor + width // 2)
    y2d = int(-y * factor + height // 2)
    return x2d, y2d

def create_matrix_points(n_points:int,radius:float,noise_bias:float = 0) -> np.array: 
    M = []

    for _ in range(n_points):
        M.append(create_random_point(radius,noise_bias))

    return np.array(M)


def create_microfone_object(config):
    

    FORMAT = pyaudio.paInt16 
    CHANNELS = 1 
    RATE = 16000 
    CHUNK = config["target_length"]
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )


    return stream,audio

def is_silence(data, threshold=300):
    rms = np.sqrt(np.mean(data**2))
    return rms < threshold

def process_micro(q, config: dict, model):
    stream_object,audio = create_microfone_object(config)
    while True:
        
        data = stream_object.read(config["target_length"], exception_on_overflow=False)
        data = np.frombuffer(data, dtype=np.int32).astype(np.float32)
        waveform = np.nan_to_num(data,nan=0).copy()

        mask = enveloppe(waveform, 16000)
        waveform = waveform[mask]
        waveform = pad_waveform(waveform, config["target_length"])

        test = waveform / 32768.0
        if is_silence(test):
            print("silence")
            continue
        
        
        harmonic, percussive = librosa.effects.hpss(waveform)
        waveform = harmonic



        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=16000,
            n_fft=config["nfft"],
            hop_length=config["hop_length"],
            n_mels=config["nmels"]
        )

        
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        x = mel_spectrogram
        x_min = x.min()
        x_max = x.max()

        x = (x - x_min) / (x_max - x_min)
        x = x.reshape(1, x.shape[0], x.shape[1], 1)

        

        res = model.predict(x)

        print(res)

        q.put(np.argmax(res))




if __name__ == "__main__":

    config = import_config("functions/config.toml")
    model = mlflow.pyfunc.load_model(
        "models:/m-84367e9851634dcfb484ce4d1742e2e8"
    )
    print(model)
    q = Queue()
    p = Process(target=process_micro,args=(q,config,model),daemon=True)
    p.start()


    pygame.init()

    WIDTH = 800
    HEIGHT = 600
    RADIUS = 2
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    theta = 0


    M = create_matrix_points(200,RADIUS,0.01)
    proj = partial(project, width=WIDTH, height=HEIGHT)

    while running: 
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                p.join()
                running = False

        while not q.empty():
            label = q.get()

            match label:
                case 0:
                    print("OLE")
            
        theta+=0.1

        R = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0,  np.cos(theta)]
            ])
        
        MR =  np.dot(M, R.T)

    
        result = map(proj,MR)

        for x, y in result:
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 2)

        pygame.display.flip()
        clock.tick(20)

        



