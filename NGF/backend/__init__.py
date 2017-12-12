import keras
if keras.backend.backend() == 'theano':
    from .theano_backend import *

else:
    raise ImportError("Tensorflow not currently supported")
