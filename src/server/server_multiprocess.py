# server.py
import logging
from utils_2.logging_level import LOGGING_LEVEL
logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('server')
log.setLevel(LOGGING_LEVEL)

from detector import Detector_Yolov5, Render
from agent import AgentDDPG, AgentEnv
from utils_2.config_parser import ConfigReader

if __name__ == '__main__':
    conf = ConfigReader().get_params()

    target = 'person'
    detector = Detector_Yolov5(conf=conf, target=target)

    render = Render(conf=conf, classes=detector.get_labels)

    input_dims = (2,)
    n_actions = 7
    env = AgentEnv(input_dims, n_actions)
    agent = AgentDDPG(conf=conf, env=env)

    detector.start()
    render.start()
    agent.start()
