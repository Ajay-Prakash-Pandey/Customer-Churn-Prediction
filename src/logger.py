import logging
import os
from datetime import datetime
from pathlib import Path

from config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    Path(LOG_DIR),
    f"{datetime.now().strftime('%Y-%m-%d')}.log"
)

logging.basicConfig(
    filename=LOG_FILE,
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
    filemode="a"
)

logger = logging.getLogger(__name__)
