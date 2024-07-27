from enum import Enum

class ResponseSignal(Enum) :

    FILE_VALIDATE_SUCCESS = "file_validate_successfully" 
    FILE_VALIDATE_FAIL =  "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded_the_allowed_size"
    FILE_UPLOAD_FAILED = 'file_upload_failed'