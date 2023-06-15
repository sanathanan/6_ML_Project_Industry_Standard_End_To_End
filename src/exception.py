import sys
import logging
from src.logger import logging

"""
This function will get "error" and display the error message using 'sys' module.
'sys' will know about each error.
"""
def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error Occured in python script name [{0}] line number [{1}] erro message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message


"""
Creating custom exception class inheriting from the Exception.
"""
class CustomException(Exception):
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message


        