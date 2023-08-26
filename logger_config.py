# logger_config.py
import logging
from termcolor import colored

class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        'INFO': 'yellow',
        'DEBUG': 'grey',
        'WARNING': 'magenta',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    LINE_COLORS = {
        'INFO': 'green',
        'DEBUG': 'green',
        'WARNING': 'green',
        'ERROR': 'green',
        'CRITICAL': 'green',
    }

    FILENAME_COLOR = 'yellow'

    def format(self, record):
        log_message = super().format(record)
        line_color = self.LINE_COLORS.get(record.levelname, 'magenta')
        colored_message = colored(log_message, line_color)
        
        levelname_color = self.LEVEL_COLORS.get(record.levelname, 'magenta')
        colored_levelname = colored(record.levelname, levelname_color, attrs=['bold'])
        
        colored_filename = colored(record.filename, self.FILENAME_COLOR, attrs=['bold'])
        
        colored_message = colored_message.replace(record.levelname, colored_levelname)
        return colored_message.replace(record.filename, colored_filename)

# Create and configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = ColoredFormatter('%(asctime)s - [%(levelname)s] - [%(filename)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

logger.addHandler(handler)
