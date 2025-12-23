import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message: Exception):
        super().__init__(error_message)
        self.error_message = CustomException._get_detailed_error(error_message)

    @staticmethod
    def _get_detailed_error(error: Exception) -> str:
        _, _, tb = sys.exc_info()
        file_name = tb.tb_frame.f_code.co_filename
        line_number = tb.tb_lineno
        return f"Error in {file_name} at line {line_number}: {str(error)}"

    def __str__(self):
        return self.error_message
