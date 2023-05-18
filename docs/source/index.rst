TMBot Documentation
==================================================================

**TMBot** is a Reinforcement Learning (RL) package for Nadeo's Trackmania 2020 `Trackmania 2020 <https://www.ubisoft.com/en-us/game/trackmania/trackmania>`_.
It is built off `Stable Baselines 3 <https://github.com/DLR-RM/stable-baselines3>`_ and aims to provide a painless interface for testing and developing
new RL methods in customizable 3D environments. 

**TMBot** uses its supplementary plugin, **TMData**, to communicate with the Trackmania process using the `OpenPlanet <https://openplanet.dev/>`_ API.
The documentation for **TMData** is included here as well.

Due to the limitations of using proprietary software as a RL environment, only a handful of data can be collected directly from the
process itself. This, along with the game's modern graphics and physics engine, can provide a more realistic and less biased
environment than most commonly used alternatives, such as `Atari environments <https://gymnasium.farama.org/environments/atari>`_.

|

.. toctree::
   :maxdepth: 3
   :caption: User Guide

   tutorial/start
   tutorial/basic

|

.. toctree::
   :maxdepth: 3
   :caption: API

   api/tmbaseenv
   api/tmgui
   api/io
   api/util

|

.. toctree::
   :maxdepth: 1
   :caption: Extras

   extras/callbacks
   extras/extractors

|

.. toctree::
   :maxdepth: 3
   :caption: TMData

   tmdata/usage

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
