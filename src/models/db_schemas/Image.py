from pydantic import BaseModel , Field , validator
from bson.objectid import ObjectId
from typing import Optional

class Image(BaseModel) :
    id : Optional[ObjectId] = Field(None , alias = "_id")
    image_path : str
    project_id : ObjectId
    
    class Config :
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls) :
        return [
            {
                "keys" : [
                    ("project_id",1)
                ],
                "name" : "project_id_index" ,
                "unique" : False 
            }
        ]