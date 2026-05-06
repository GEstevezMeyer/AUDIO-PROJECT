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


def create_lines_effect(amount_lines:int,height,width):
    res = []
    for i in range(amount_lines): 
        start = (np.random.uniform(0,width),0)
        end = (np.random.uniform(0,width),height)
        res.append((start,end))

    return res
        

def clean_x(x:np.ndarray) -> np.ndarray:

    x = (x-x.min())/(x.max()-x.min())
    x = x.reshape(1,x.shape[0],x.shape[1],1)

    return x 

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
        frames_per_buffer=CHUNK,
        input_device_index=2
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

        

        x1 = clean_x(mel_spectrogram)
       

        if config["model"] == "double":
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=16000,
                n_mfcc=config["nmels"]
            )

            x2 = clean_x(mfcc)
            x = (x1,x2)
        
        else:
            x = x1

        res = model.predict(x)

        print(res)

        q.put(np.argmax(res))

def is_in_the_circle(x,y,center,radius): 
    xc , yc = center
    return (x-xc)**2 + (y-yc)**2 <= radius**2




if __name__ == "__main__":
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        print("INDEX:", i)
        print("NAME :", info["name"])
        print("INPUT CHANNELS:", info["maxInputChannels"])
        print("OUTPUT CHANNELS:", info["maxOutputChannels"])
        print()

    mlflow.set_tracking_uri("http://localhost:5000")

    config = import_config("functions/config.toml")
    model = mlflow.pyfunc.load_model(
        "models:/m-030f7cc2b8884929befdd77e67199b61"
    )

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

    info = pygame.display.Info()
   
    WIDTH = info.current_w
    HEIGHT = info.current_h

    CENTER = (WIDTH//2, HEIGHT//2)

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    running = True
    
    
    GameColors = Colors((0, 0, 0),(40, 40, 40),(0,200,0),(255,255,255),q_tempos_color)
    RotationMatrix = Rotation_Matrix(q_tempos_rotation,clock)
    Proj = Projector(WIDTH,HEIGHT)
    M = create_matrix_points(100,RADIUS,0.01)

    generator_rgb_screen = generator_rgb_effect(GameColors.screen,GameColors.targetScreen,20)
    generator_rgb_dots = generator_rgb_effect(GameColors.dots,GameColors.targetDots,20)

    state_drawing = "circle"
    n_lines = 4
    
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

        pygame.draw.circle(screen,np.array(screen_color)*0.5, CENTER ,40)


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
                    state_drawing = "circle"
                    if n_lines <= 20:
                        n_lines+= 5
                case 6:
                    state_drawing = "rectangle"
                    if n_lines >= 10:
                        n_lines-= 5
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

        if flashScreenFlag:
            effect_thunder = create_lines_effect(n_lines,HEIGHT,WIDTH)
    
            for start_position,end_position in effect_thunder: 
                pygame.draw.line(screen,np.array(dots_color)*0.5, start_position,end_position, 1)

        for x, y in result:
            if is_in_the_circle(x,y,CENTER,40):
                continue

            if flashScreenFlag:
                if state_drawing == "circle":
                    pygame.draw.circle(screen,np.array(dots_color)*0.25, (x, y), 2)
                elif state_drawing == "rectangle": 
                    pygame.draw.rect(screen,np.array(dots_color)*0.25 , (x, y,2,2))
            else:
                if state_drawing == "circle":
                    pygame.draw.circle(screen,dots_color, (x, y),2)
                elif state_drawing == "rectangle": 
                    pygame.draw.rect(screen,dots_color,(x,y,2,2))



        pygame.display.flip()
        clock.tick(FRAME_LIMITS)

        



