import sys
from typing import Optional


def error_message_detail(error: Exception, error_detail: Optional[object] = None) -> str:
    """
    Generate a detailed error message including
    file name and line number.
    """

    if error_detail is None:
        return str(error)

    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in Python script\n"
        f"File Name : {file_name}\n"
        f"Line Number : {line_number}\n"
        f"Error Message : {str(error)}"
    )


class CustomException(Exception):
    """
    Custom exception class for the project.
    """

    def __init__(self, error: Exception, error_detail: Optional[object] = None):

        super().__init__(str(error))

        self.error_message = error_message_detail(
            error,
            error_detail
        )

    def __str__(self):

        return self.error_message