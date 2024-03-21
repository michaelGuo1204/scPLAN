import logging
import sys

logger = logging.getLogger("PLL Trainer")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)
