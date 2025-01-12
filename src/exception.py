import sys
from src.logger import logging

def error_message_details(error, error_detail:sys):
    _, _, tb = sys.exc_info()
    error_message = "Error: " + str(error) + " at line " + str(tb.tb_lineno) + " in " + str(tb.tb_frame.f_code.co_filename)
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

# if __name__ == '__main__':
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Division by zero")
#         raise CustomException(e, sys)