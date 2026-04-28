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

class Rotation_Matrix:
    def __init__(self):
        self._theta = 0.0
        self._inverse = False
        self._amount = 0.05

    @property
    def matrix(self):
        R = np.array([
            [np.cos(self._theta), 0, -np.sin(self._theta)],
            [0, 1, 0],
            [np.sin(self._theta), 0,  np.cos(self._theta)]
        ])

        return R.T if self._inverse else R


    def add_theta(self):
        self._theta += self._amount


    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = float(value)


    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, value):
        if 0 < value < np.pi:
            self._amount = float(value)


    @property
    def inverse(self):
        return self._inverse

    @inverse.setter
    def inverse(self, value):
        self._inverse = bool(value)

class Colors:
    def __init__(self):
        self._screen = [0, 0, 0]
        self._dots = [255, 255, 255]

    @property
    def screen(self):
        return tuple(self._screen)

    @screen.setter
    def screen(self, new_screen: list):
        self._screen = self._fix_rgb(new_screen)

    @property
    def dots(self):
        return tuple(self._dots)

    @dots.setter
    def dots(self, new_dots: list):
        self._dots = self.fix_rgb(new_dots)

    @staticmethod
    def fix_rgb(values):
        for i in range(3):
            if values[i] < 0:
                values[i] = 255
            elif values[i] > 255:
                values[i] = 0

        return values

    

class Projector():
    def __init__(self,width,height):
        self._width = width
        self._height = height
        self._scale = 200
        self._d = 3

    def project(self,point):
        x, y, z = point
        factor = self.scale / (z + self.d)
        x2d = int(x * factor + self.width // 2)
        y2d = int(-y * factor + self.height // 2)
        return x2d, y2d
    

    @property
    def scale(self):
        return self._scale

    @property
    def d(self):
        return self._d

    @scale.setter
    def scale(self,amount:float):
        if amount >= 1000:
            self._scale = 300
        elif amount <= 0:
            self._scale = 700
        
        self._scale = amount

    @d.setter
    def d(self,amount):
        if amount <= 0 or amount >= 13:
            self._d = 3

        self._d = 3

    @property
    def width(self): 
        return self._width
    
    @property
    def height(self):
        return self._height


    
    

    

def create_random_point(radius:float,noise_bias:float = 0.05) -> tuple: 
    z = np.random.uniform(-radius,radius)
    theta = np.random.uniform(0,2*np.pi)
    x = np.sqrt(radius**2-z**2)*np.cos(theta)+np.random.normal(0,noise_bias)
    y = np.sqrt(radius**2-z**2)*np.sin(theta)+np.random.normal(0,noise_bias)

    return (x,y,z)


def create_matrix_points(n_points:int,radius:float,noise_bias:float = 0) -> np.array: 
    M = []

    for _ in range(n_points):
        M.append(create_random_point(radius,noise_bias))

    return np.array(M)

def add_point(M,RADIUS:float):
    res = M.tolist()
    for _ in range(5):
        res.append(create_random_point(RADIUS,0.8))
    
    return np.array(res)

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

def is_silence(data, threshold=200):
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
    

    GameColors = Colors()
    RotationMatrix = Rotation_Matrix()
    Proj = Projector(WIDTH,HEIGHT)

    M = create_matrix_points(200,RADIUS,0.01)
    
    while running: 
        screen.fill(GameColors.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                p.join()
                running = False

        while not q.empty():
            label = q.get()

            match label:
                case 0:
                    M = add_point(M,RADIUS)
                case 1:
                    Proj.d+=0.1
                    M = add_point(M,RADIUS)
                case 2:
                    RotationMatrix.inverse = not(RotationMatrix.inverse)
                    M = add_point(M,RADIUS)
                case 3:
                    Proj.d-=0.1
                    M = add_point(M,RADIUS)
                case 4:
                    Proj.scale-=100
                    M = add_point(M,RADIUS)
                case 5: 
                    RotationMatrix.theta+=0.1
                case 6:
                    RotationMatrix.theta-=0.1
                case 7:
                    Proj.scale+=100
                    M = add_point(M,RADIUS)


        RotationMatrix.add_theta()

        R = RotationMatrix.matrix
        MR =  np.dot(M, R.T)

    
        result = map(Proj.project,MR)

        for x, y in result:
            pygame.draw.circle(screen, GameColors.dots, (x, y), 2)

        pygame.display.flip()
        clock.tick(20)

        



