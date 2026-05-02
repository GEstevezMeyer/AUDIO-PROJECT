import numpy as np 
from multiprocessing import Queue


class Rotation_Matrix:
    def __init__(self,q_tempos:Queue,clock):
        self._theta = 0.0
        self._inverse = False
        self._amount = 0.05
        self._queue = q_tempos
        self._k = 0.2  
        self._clock = clock
        self._tempo = [128]

    @property
    def matrix(self):
        R = np.array([
            [np.cos(self._theta), 0, -np.sin(self._theta)],
            [0, 1, 0],
            [np.sin(self._theta), 0,  np.cos(self._theta)]
        ])

        return R.T if self._inverse else R


    def add_theta(self):
        self.theta += self.amount

    def update_amount(self):
        
        if not self._queue.empty():
            x,_ = self._queue.get()
            self._tempo.append(np.squeeze(x))
            fps = self._clock.get_fps()
            if fps < 20:
                fps = 20

            self.amount = ((np.pi*2*self.tempoMean)/(60*fps))*self._k

            



    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = float(value)
    
    @property
    def tempo(self):
        return self._tempo
    
    @tempo.setter
    def tempo(self,value):
        self._tempo = value

    @property
    def amount(self):
        return self._amount
    
    @amount.setter
    def amount(self,value):
        self._amount = value


    @property
    def inverse(self):
        return self._inverse

    @inverse.setter
    def inverse(self, value):
        self._inverse = bool(value)

    @property
    def tempoMean(self):
        return np.mean(self.tempo[-4:])

class Colors:
    def __init__(self,screen,targetScreen,dots,targetDots,q_tempos:Queue):
        self._screen = screen
        self._targetScreen = targetScreen
        self._flash = (255,255,255)
        self._queue = q_tempos
        self._tempo = [60/128]
        self._dots = dots
        self._targetDots = targetDots

    @property
    def targetScreen(self):
        return self._targetScreen
    
    @targetScreen.setter
    def targetScreen(self,new_screen:list):
        self._targetScreen = tuple(self.fix_rgb(new_screen))

    @property
    def targetDots(self):
        return self._targetDots
    
    @targetDots.setter
    def targetDots(self,new_screen:list):
        self._targetDots = tuple(self.fix_rgb(new_screen))

    @property
    def screen(self):
        return tuple(self._screen)

    @screen.setter
    def screen(self, new_screen: list):
        self._screen = tuple(self.fix_rgb(new_screen))

    @property
    def dots(self):
        return tuple(self._dots)

    @dots.setter
    def dots(self, new_screen: list):
        self._dots = tuple(self.fix_rgb(new_screen))

    @property
    def flash(self):
        return tuple(self._flash)
    
    @flash.setter
    def flash(self, new_dots: list):
        self._flash = tuple(self.fix_rgb(new_dots))

    @property
    def tempo(self):
        return self._tempo
    
    @tempo.setter
    def tempo(self,value):
        self._tempo = value

    @property
    def tempoMean(self):
        return np.mean(self.tempo[-4:])

    @staticmethod
    def fix_rgb(values):
        for i in range(3):
            if values[i] < 0:
                values[i] = 255
            elif values[i] > 255:
                values[i] = 0

        return values

    def flash_screen(self,value):

        if self.tempoMean<= value:
            return True
        else:
            return False
    
    def update_tempo(self):
        if not self._queue.empty():
            x,_ = self._queue.get()
            x = float(np.squeeze(x))
            self._tempo.append(60/x)
            


class Projector():
    def __init__(self,width,height):
        self._width = width
        self._height = height
        self._scale = 200
        self._d = 3
        self._targetScale = 200

    def project(self,point:list) -> tuple:
        x, y, z = point
        factor = self.scale / (z + self.d)
        x2d = int(x * factor + self.width // 2)
        y2d = int(-y * factor + self.height // 2)
        return x2d, y2d
    
    def update_scale(self):
        if self.scale != self.targetScale:
            if self.scale > self.targetScale:
                self.scale-= 20
            else:
                self.scale+= 20
    

    @property
    def targetScale(self):
        return self._targetScale
    
    @targetScale.setter
    def targetScale(self,amount): 
        if amount >= 1000 or amount <= 200:
            self._targetScale = 300
        else:
            self._targetScale = amount

    @property
    def scale(self):
        return self._scale

    @property
    def d(self):
        return self._d

    @scale.setter
    def scale(self,amount:float):
        if amount >= 1000 or amount <= 200:
            self._scale = 300
        else:
            self._scale = amount

    @d.setter
    def d(self,amount):
        if amount <= 0 or amount >= 13:
            self._d = 3

        self._d = amount

    @property
    def width(self): 
        return self._width
    
    @property
    def height(self):
        return self._height