# -- coding: utf-8 --
import os
import sys
import time
import logging
import colorlog

from utils.env import Env
from utils.file import prepare_directory

def _configure_logger(as_stream=True, as_file=False):
    logger = logging.getLogger("default-logger")
    logger.setLevel(logging.DEBUG)

    if as_stream:
        format = colorlog.ColoredFormatter(
            fmt="%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG':'gray',
                'INFO':"white",
                'WARNING':'yellow',
                'ERROR':'red',
            },
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(format)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    if as_file:
        file_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + ".log"
        file_dir = prepare_directory(Env.LOG_DIR)
        file_path = os.path.join(file_dir, file_name)
        print(file_path)
        file_handler = logging.FileHandler(file_path, "a", encoding="utf-8")
        format = logging.Formatter(
            fmt="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(format)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger

logger = _configure_logger(True, True)


if __name__ == "__main__":
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")