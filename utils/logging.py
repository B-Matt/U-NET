import logging
import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

# Edit Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Removes default logger handlers
for handler in list(logger.handlers):
    logger.removeHandler(handler)

# logfmt_str = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)03d:%(funcName)s %(message)s"
# formatter = logging.Formatter(logfmt_str)

logger.addHandler(TqdmLoggingHandler())

"""streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG)
logger.addHandler(streamHandler)"""