from base import TMBaseEnv
from callbacks import TMPauseOnUpdate, TMSaveOnEpoch, TMResetOnEpoch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList

op_path = r"C:\Users\Lab\OpenplanetNext"
frame_shape = (3, 64, 64) # Channels, height, width

policy_kwargs = dict(
    net_arch=[256] * 4
)

# Max framerate (sleep between writing action and getting observation)
# Divide timestep reward by framerate (max framerate?)

# simple CNN, 3 layers, (256, 512, 512), to 512 obs features

if __name__ == "__main__":
    env = TMBaseEnv(frame_shape = frame_shape, gui = True)
    model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=1, n_steps=2048, n_epochs=50, policy_kwargs=policy_kwargs) # TODO: Hyperparameter optimization
    callbacks = CallbackList([TMPauseOnUpdate(), TMResetOnEpoch(), TMSaveOnEpoch("model/TMTest")])

    model = model.load("model/TMTest_19", env=env)

    model.learn(total_timesteps=100000, callback=callbacks, progress_bar=True)
