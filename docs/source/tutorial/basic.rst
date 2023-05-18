Basic Usage
================

Creating an Environment
-----------------------

Most **TMBot** functionality is handled automatically by :doc:`../api/tmbaseenv`. It's usage is as simple as

.. code-block:: python

   from tmbot.core import TMBaseEnv

   kwargs = dict()
   env = TMBaseEnv(**kwargs)

The usage of :doc:`../api/tmbaseenv` is covered heavily in its documentation, but some key points will be briefly touched upon here.

 - Specifying the correct `op_path` is required if it's anything other than the default
 - The observation `frame_shape` controls the resolution and color space of visual input
 - Although specific observations can be disabled for the environment, those observations should be disabled in the **TMData** settings as well.
 - :doc:`../api/tmgui` is helpful for troubleshooting and visualizing both the model and environment
 - The `reward` function is intended to be overridden for alternate implementation
 - Knowledge of `gymnasium <https://gymnasium.farama.org/>`_ isn't necessary 

Creating a Model
----------------

**TMBot** uses the `Stable Baselines 3 <https://github.com/DLR-RM/stable-baselines3>`_ backend for its models.
Using this, a simple model can be created with

.. code-block:: python

   from sb3_contrib import RecurrentPPO

   kwargs = dict()
   model = RecurrentPPO(policy="MultiInputLstmPolicy", env=env, **kwargs)

Any `stable_baselines3` algorithm supporting a `MultiDiscrete` action space is compatible with **TMBot**. At least a basic knowledge of
machine learning and reinforcement learning techniques is helpful for finding the most optimal implementations for a desired use case.

Additionally **TMBot** includes the `custom_extractor_policy` function to automate the creation of a custom features extractor.

.. code-block:: python

   from tmbot.extras.extractors import custom_extractor_policy, SimpleCNN

   model_kwargs = dict()
   policy_kwargs = custom_extractor_policy(model=SimpleCNN, model_kwargs=model_kwargs)

of which additional arguments can be added onto with

.. code-block:: python

   policy_kwargs["net_arch"] = [128, 64, 64]

Callbacks
----------

Callbacks are functions called at certain times during the training process, some of which are near essential for proper training.
To keep it simple, `default_callbacks` can be used.

.. code-block:: python

   from tmbot.extras.callbacks import default_callbacks

   callbacks = default_callbacks()

Training
---------

Before training, it is important to note that training a model on Trackmania will take longer than on most other environments,
and substantially longer than on traditional datasets. This is because only one instance of Trackmania can be run at a time without
the usage of an ancient form of black magic known as *"virtual machines"*.

Instead of wrapping an existing game, most commonly used `gymnasium` environments directly host an environments code within the
python script, which allows *n* identical environments to run simultaneously while gathing *n* times more data.

With this in mind, the standard `stable_baselines3` training procedure can be used.

.. code-block:: python

   model.learn(total_timesteps=1e5, callback=callbacks, progress_bar=True)

   model.save("model")

Evaluation
-----------

**TODO**
