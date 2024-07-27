from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignal
import re
from .ProjectController import ProjectController 
import os


class DataController(BaseController) :
    
    def __init__(self) :
        super().__init__()
        self.scale_size = 1000000 # from MB to byte

    
    def validate_uploaded_data(self , file : UploadFile ) :

        if file.content_type not in self.app_settings.ACCEPTED_FILE_TYPES :

            return False , ResponseSignal.FILE_VALIDATE_FAIL.value
        
        if file.size > self.app_settings.MAX_INPUT_SIZE * self.scale_size :

            return False , ResponseSignal.FILE_SIZE_EXCEEDED.value
        

        return True , ResponseSignal.FILE_VALIDATE_SUCCESS.value
    
    def get_clean_file_name(self, original_file_name: str):


        cleaned_file_name = re.sub(r'[^\w.]', '', original_file_name.strip())

        cleaned_file_name = cleaned_file_name.replace(" ", "_")

        return cleaned_file_name
    
    def get_unique_file_path(self , project_id : str , original_file_name : str ) :
        
        clean_original_file_name = self.get_clean_file_name(original_file_name = original_file_name)

        random_string = self.generate_random_string()

        project_path = ProjectController().get_project_path(project_id=project_id)

        new_file_path = os.path.join( project_path , random_string + "_" + clean_original_file_name )

        while os.path.exists(new_file_path) :
            random_string = self.generate_random_string()
            new_file_path = os.path.join( project_path , random_string + "_" + clean_original_file_name )

        return new_file_path , random_string + "_" + clean_original_file_name