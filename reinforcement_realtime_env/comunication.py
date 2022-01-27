
import pickle
import logging

logging.basicConfig(level=logging.INFO)
HEADERSIZE = 10

def receive(socket) -> str:
    got_full_msg = False
    is_new_msg = True
    full_msg = b''
    while not got_full_msg:
        part_msg = socket.recv(16)
        if is_new_msg:
            len_msg = int(part_msg[:HEADERSIZE])
            is_new_msg = False
            logging.debug(f"got new message length: {len_msg}")

        full_msg += part_msg

        if len(full_msg) - HEADERSIZE == len_msg:
            msg = full_msg[HEADERSIZE:]
            got_full_msg = True

    msg = pickle.loads(msg)
    logging.debug(f'full msg received: {msg}')
    return msg


def send(socket, msg) -> None:
    msg = pickle.dumps(msg)
    msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
    socket.send(msg)
