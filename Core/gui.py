import gymnasium as gym
import tkinter as tk
import numpy as np
import threading
from PIL import ImageTk, Image

# TODO: See observations, percent steps held, ts_reward and goal_reward trends

class TMGUI(threading.Thread):
    def __init__(self, env : gym.Env, frame_size : tuple[int, int]):
        threading.Thread.__init__(self)
        self.start()

        self.env = env
        self.frame_size = frame_size

        self.window = tk.Tk()
        self.window.title("TMBot")

        frame_image = ImageTk.PhotoImage(Image.new(mode="RGB", size=env.frame_shape[1:]).resize(self.frame_size))
        self.frame_display = tk.Label(image=frame_image)
        self.frame_display.pack()

    def update(self):
        self.window.update()

    def set_frame(self, frame_image : np.ndarray):
        frame_image = Image.fromarray(frame_image).resize(self.frame_size, resample=Image.Resampling.NEAREST)
        frame_tk = ImageTk.PhotoImage(frame_image)
        self.frame_display.configure(image=frame_tk)
        self.frame_display.image = frame_tk
