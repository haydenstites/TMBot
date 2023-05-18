Getting Started
================

Installation
------------

.. note::

   **TMBot** depends on the Windows API, and can only be used on a Windows operating system.

**TMBot** can either be installed using `pip` or directly from the `GitHub <https://github.com/Hayden-Stites/TMBot>`_ repository

.. code-block:: console

   (env) $ pip install tmbot

or

.. code-block:: console

   (env) $ pip install git+https://github.com/Hayden-Stites/TMBot/src/dist/tmbot-1.0.0.tar.gz

Installing TMData
------------------

**TMBot** uses its supplementary plugin, **TMData**, to communicate with the Trackmania process using the `OpenPlanet <https://openplanet.dev/>`_ API.
Due to this, OpenPlanet must be installed prior to usage. It is recommended to install it to the default path to
avoid specifying `op_path` during training.

During the first initialization of :doc:`../api/tmbaseenv`, **TMData** should automatically download to the environment's specified
`op_path` and pause execution. If this doesn't work, or you want to fully install everything before usage, either run 
`init_tmdata.py <https://github.com/Hayden-Stites/TMBot/blob/master/init_tmdata.py>`_ from the console

.. code-block:: console

   (env) $ python init_tmdata.py -o OPENPLANET_PATH

or directly run

.. code-block:: python

   from tmbot.core import init_tmdata
   init_tmdata(op_path=None)

to download and install **TMData** and its dependencies and an OpenPlanet plugin.

Running The Demo
-----------------

The demo file `demo.py <https://github.com/Hayden-Stites/TMBot/blob/master/demo.py>`_ is available on
the `GitHub <https://github.com/Hayden-Stites/TMBot>`_ repository and shows off basic usage of the **TMBot** package.

To test out your installation, its a good idea to test out a pre-trained model. These can be found **HERE**
You can either directly invoke the `predict_demo` method, or simply run it through the console

.. code-block:: console

   (env) $ python demo.py -p -m MODEL_PATH

.. note::

   If OpenPlanet is installed somewhere other than the default location, run `demo.py` with the `-o` argument to specify your install location.

If you have multiple displays, you can pass the `-g` argument to see what the model sees. During training, simply switch to the Trackmania process and watch your model play!

For further reading, make sure to check out :doc:`basic`.
