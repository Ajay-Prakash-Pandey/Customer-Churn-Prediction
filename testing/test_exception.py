import sys

from src.exception import CustomException
from src.logger import logger


def divide_numbers():

    try:

        logger.info("Division Started")

        result = 10 / 0

        return result

    except Exception as e:

        logger.error("Division Failed")

        raise CustomException(e, sys)


if __name__ == "__main__":

    divide_numbers()