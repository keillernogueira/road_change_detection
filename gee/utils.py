from __future__ import print_function
from datetime import datetime
import logging
import time
import os

from pytz import timezone, utc


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.error("default")

    logging.Formatter.converter = time.localtime
    logger.error("localtime")

    logging.Formatter.converter = time.gmtime
    logger.error("gmtime")

    def custom_time():
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("America/Sao_Paulo")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    logging.Formatter.converter = custom_time
    logger.error("customTime")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


if __name__ == '__main__':
    pass
