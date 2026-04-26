import numpy as np 
import matplotlib.pyplot as plt 
import pygame
from multiprocessing import Pool,cpu_count
from functools import partial
import pyaudio
from training_model import import_config



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


def create_microfone_object(config_path:str = "functions/config.toml"):
    config = import_config(config_path)

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
            running = False

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

    



