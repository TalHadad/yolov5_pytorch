from typing import Any
import zlib
import pickle

def send(socket, msg: Any) -> None:
    """pickle an object, and zip the pickle before sending it"""

    p = pickle.dumps(msg, protocol=-1)
    z = zlib.compress(p)
    socket.send(z, flags=0)


def receive(socket) -> Any:
    """inverse of send_zipped_pickle"""
    z = socket.recv(copy=False, flags=0)
    p = zlib.decompress(z)
    msg = pickle.loads(p)
    return msg
