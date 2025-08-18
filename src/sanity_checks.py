import inspect
import domain

from domain.beatmap import BSMap
import os, sys
import numpy as np
import librosa


print("Numpy version :" + np.__version__)

print(" savez compressed signatur", inspect.signature(np.savez_compressed))

print("func repr:", np.savez_compressed)
# print(    "normalize :",    normalize.__module__,    normalize.__qualname__,    inspect.signature(normalize),)
# print("normalize annotation:", normalize.__annotations__)

# try:
#    np.savez_compressed(".cache/should_fail.npz", {"rms": np.array(0.5)})
#    print("Unexpected : dict positionnel n'a pas échoué comme prévu ")
# except Exception as e:
#    print("OK dict positionnel a échoué comme prévu :", type(e).__name__, e)

safe = {
    "__meta__": np.array(b"{}"),
    "rms": np.array(0.5),
    "vec": np.array([1.0, 2.0, 3.0]),
}
try:
    np.savez_compressed("test.npz", **safe, allow_pickle=False)
    print("OK dict unpacking a fonctionné comme prévu")
except Exception as e:
    print("Unexpected : dict unpacking a échoué", type(e).__name__, e)
