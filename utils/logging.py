import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Removes default logger handlers
for handler in list(logger.handlers):
    logger.removeHandler(handler)

logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(logfmt_str)

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG)

logger.addHandler(streamHandler)