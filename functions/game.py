import numpy as np 
import pygame
from multiprocessing import Process,Queue
import pyaudio
from training_model import import_config
import librosa
import mlflow.pyfunc
from audio_pipeline import enveloppe,pad_waveform
from classes import *


def generator_rgb_effect(start_color:list , end_color:list,steps:int):
    r = np.linspace(start_color[0],end_color[0],steps)
    g = np.linspace(start_color[1],end_color[1],steps)
    b = np.linspace(start_color[2],end_color[2],steps)
    for i in range(steps):
        yield (r[i],g[i],b[i])


    

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

def process_micro(q,q_tempos_color,q_tempos_rotation,config: dict, model):
    stream_object,audio = create_microfone_object(config)
    while True:
        
        data = stream_object.read(config["target_length"], exception_on_overflow=False)
        data = np.frombuffer(data, dtype=np.int32).astype(np.float32)
        waveform = np.nan_to_num(data,nan=0).copy()

        tempo, beat_times = librosa.beat.beat_track(y=waveform,sr = 16000,hop_length=config["hop_length"])


        q_tempos_color.put((tempo,beat_times))
        q_tempos_rotation.put((tempo,beat_times))

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

        q.put(np.argmax(res))




if __name__ == "__main__":

    mlflow.set_tracking_uri("http://localhost:5000")

    config = import_config("functions/config.toml")
    model = mlflow.pyfunc.load_model(
        "models:/m-84367e9851634dcfb484ce4d1742e2e8"
    )

    WIDTH = 800
    HEIGHT = 600
    RADIUS = 2
    FRAME_LIMITS = 20
    DT = 1/FRAME_LIMITS
    time = 0
 
    q = Queue()
    q_tempos_color = Queue()
    q_tempos_rotation = Queue()
    p = Process(target=process_micro,args=(q,q_tempos_color,q_tempos_rotation,config,model),daemon=True)
    p.start()


    pygame.init()
   
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    running = True
    
    
    GameColors = Colors((0, 0, 0),(40, 40, 40),(200,200,200),(255,255,255),q_tempos_color)
    RotationMatrix = Rotation_Matrix(q_tempos_rotation,clock)
    Proj = Projector(WIDTH,HEIGHT)
    M = create_matrix_points(100,RADIUS,0.01)

    generator_rgb_screen = generator_rgb_effect(GameColors.screen,GameColors.targetScreen,20)
    generator_rgb_dots = generator_rgb_effect(GameColors.dots,GameColors.targetDots,20)

    
    while running: 
        time+= DT
        flashScreenFlag = GameColors.flash_screen(time)

        

        if flashScreenFlag:
            time-= GameColors.tempoMean

        try:
            screen_color = next(generator_rgb_screen)
        except StopIteration:
            GameColors.screen , GameColors.targetScreen = GameColors.targetScreen,GameColors.screen
            generator_rgb_screen = generator_rgb_effect(GameColors.screen,GameColors.targetScreen,20)
            screen_color = next(generator_rgb_screen)

        try:
            dots_color = next(generator_rgb_dots)
        except StopIteration:
            GameColors.dots , GameColors.targetDots = GameColors.targetDots,GameColors.dots
            generator_rgb_dots = generator_rgb_effect(GameColors.dots,GameColors.targetDots,20)
            dots_color = next(generator_rgb_dots)

        


        
        screen.fill(screen_color)

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
                    Proj.targetScale-=200
                    M = add_point(M,RADIUS)
                case 5: 
                    M = create_matrix_points(200,RADIUS)
                case 6:
                    M = create_matrix_points(100,RADIUS)
                case 7:
                    Proj.targetScale+=200
                    M = add_point(M,RADIUS)


        Proj.update_scale()
        GameColors.update_tempo()
        RotationMatrix.update_amount()
        RotationMatrix.add_theta()

        R = RotationMatrix.matrix
        MR =  np.dot(M, R.T)

    
        result = map(Proj.project,MR)

        for x, y in result:
            if flashScreenFlag:
                pygame.draw.circle(screen,np.array(dots_color)*0.5, (x, y), 2)
            else:
                pygame.draw.circle(screen,dots_color, (x, y), 2)

        pygame.display.flip()
        clock.tick(FRAME_LIMITS)

        



