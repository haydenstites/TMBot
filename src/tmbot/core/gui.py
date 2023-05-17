import gymnasium as gym
import tkinter as tk
import numpy as np
from PIL import ImageTk, Image

# TODO: percent steps held

class TMGUI():
    def __init__(self, enabled : dict[str, bool], rew_enabled : dict[str, bool], frame_size : tuple[int, int] = (500, 500), buffer_size : int = 30, env : gym.Env = None):
        r"""Initialization parameters for TMGUI. TMBaseEnv automatically wraps this.

        Args:
            enabled (dict[str, bool]) : Dictionary describing enabled parameters in observation space.

            rew_enabled (dict[str, bool]) : Dictionary describing enabled parameters for reward shaping.

            frame_size (tuple[int, int]) : Displayed frame size in pixels. Default is (500, 500).

            buffer_size (int) : Buffer steps for rewards to smooth displayed reward values. Default is 30 (steps).

            env (gym.Env, Optional) : A gymnasium environment that must be assigned if linked to a :class:`TMBaseEnv` object.

        """
        self.enabled = enabled
        self.rew_enabled = rew_enabled
        self.frame_size = frame_size
        self.buffer_size = buffer_size
        self.env = env
        self.uns = {}

        if type(self.buffer_size) is int and self.buffer_size > 0:
            self.uns.setdefault("ts_buffer", [])
            self.uns.setdefault("goal_buffer", [])
        else:
            self.buffer_size = 0

        self.window = tk.Tk()
        self.window.title("TMBot GUI")

        pad = 50
        self.window.columnconfigure(index=0, weight=1, minsize=max(self.frame_size[0] + pad, 400))
        self.window.rowconfigure(index=1, weight=1, minsize=max(self.frame_size[1] + pad, 400))

        sys_font = ("System", 0)

        # bar
        self.exit = tk.Button(master=self.window, text="Close GUI", command=self._close_gui, font=sys_font)
        self.exit.grid(row=0, column=0, sticky="w")

        # frame
        if enabled["frame"]:
            frame_tk = ImageTk.PhotoImage(Image.new(mode="RGB", size=env.frame_shape[1:]).resize(self.frame_size))
            self.frame_display = tk.Label(image=frame_tk, relief=tk.RIDGE, borderwidth=5)
            self.frame_display.grid(row=1, column=0)

        # reward
        self.reward_frame = tk.Frame(master=self.window)
        self.reward_frame.grid(row=2, column=0, sticky="w", padx=20, pady=10)

        self.ts_reward = tk.Label(master=self.reward_frame, text="ts_reward: NA", font=sys_font)
        self.ts_reward.grid(row=0, column=0)
        self.goal_reward = tk.Label(master=self.reward_frame, text="goal_reward: NA", font=sys_font)
        self.goal_reward.grid(row=0, column=1, padx=20)

        # obs/extra container
        self.text_frame = tk.Frame(master=self.window, relief=tk.GROOVE, borderwidth=5)
        self.text_frame.grid(row=3, column=0, sticky="nsew")

        # obs
        self.obs_frame = tk.Frame(master=self.text_frame)
        self.obs_frame.grid(row=0, column=0, sticky=tk.N)
        self.text_frame.columnconfigure(index=0, weight=1, minsize=200)

        obs = tk.Label(master=self.obs_frame, text="Observations:", font=sys_font)
        obs.pack()

        obs_text = str()
        for key in enabled:
            if enabled[key] and key != "frame":
                obs_text += f"{key}: NA\n"
        self.obs = tk.Label(master=self.obs_frame, text=obs_text, font=sys_font)
        self.obs.pack()

        # extra
        self.extra_frame = tk.Frame(master=self.text_frame, relief=tk.GROOVE)
        self.extra_frame.grid(row=0, column=1, sticky=tk.N)
        self.text_frame.columnconfigure(index=1, weight=1, minsize=200)

        extra = tk.Label(master=self.extra_frame, text="Extra Observations:", font=sys_font)
        extra.pack()

        extra_text = str()
        for key in rew_enabled:
            if rew_enabled[key]:
                extra_text += f"{key}: NA\n"
        self.extra = tk.Label(master=self.extra_frame, text=extra_text, font=sys_font)
        self.extra.pack()

        self.window.update()

    def update(self, obs, rew_vars, ts_reward, goal_reward):
        # frame
        if obs["frame"] is not None:
            image = Image.fromarray(obs["frame"].transpose()).resize(self.frame_size, resample=Image.Resampling.NEAREST)
            frame_tk = ImageTk.PhotoImage(image)
            self.frame_display.config(image=frame_tk)
            self.frame_display.image = frame_tk

        # reward
        if self.buffer_size > 0:
            assert len(self.uns["ts_buffer"]) == len(self.uns["goal_buffer"])

            if len(self.uns["ts_buffer"]) >= self.buffer_size:
                del self.uns["ts_buffer"][0]
                del self.uns["goal_buffer"][0]
            self.uns["ts_buffer"].append(ts_reward)
            self.uns["goal_buffer"].append(goal_reward)

            ts_reward = sum(self.uns["ts_buffer"]) / len(self.uns["ts_buffer"])
            goal_reward = max(self.uns["goal_buffer"])

        self.ts_reward.config(text=f"ts_reward: {round(ts_reward, 4)}")
        self.goal_reward.config(text=f"goal_reward: {goal_reward}")

        # obs
        obs_text = str()
        for key in obs:
            if key != "frame":
                obs_text += f"{key}: {np.round(obs[key], 2)}\n"
        self.obs.config(text=obs_text)

        # extra
        extra_text = str()
        for key in rew_vars:
            extra_text += f"{key}: {rew_vars[key]}\n"
        self.extra.config(text=extra_text)

        self.window.update()

    def flush_buffers(self):
        self.uns["ts_buffer"] = []
        self.uns["goal_buffer"] = []

    def _close_gui(self):
        if self.env is not None:
            self.env.gui = False
        self.window.destroy()
