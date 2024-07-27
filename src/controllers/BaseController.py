from helpers import get_settings , Settings
import os
import random
import string

class BaseController :

    def __init__(self) :
        self.app_settings = get_settings()
        self.base_directory = os.getcwd()
        self.uploaded_files_path = os.path.join(self.base_directory , 'assets/files')

    def generate_random_string(self,length=12):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for _ in range(length))