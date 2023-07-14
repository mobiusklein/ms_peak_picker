import os

import numpy as np

def load():
    path = os.path.join(
        os.path.dirname(__file__),
        "example.npz")
    arc = np.load(path)
    return arc['x'], arc['y']