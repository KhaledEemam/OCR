from models.enums import DataBaseEnum
from models.db_schemas import Image
from .BaseDataModel import BaseDataModel
from bson.objectid import ObjectId


class ImagesDataModel(BaseDataModel) :

    def __init__(self,db_client: object ) :
        super().__init__(db_client=db_client)
        self.collection = db_client[DataBaseEnum.IMAGES_COLLECTION.value]

    @classmethod
    async def create_instance(cls , db_client : object) :
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self) :
        all_collections = await self.db_client.list_collection_names()
        if self.collection not in all_collections :
            self.collection = self.db_client[DataBaseEnum.IMAGES_COLLECTION.value]
            indexes = Image.get_indexes()
            for index in indexes :
                await self.collection.create_index(
                    keys=index["keys"] ,
                    name=index["name"] ,
                    unique=index["unique"]
                )



    async def insert_image(self, image : Image) :
        record = await self.collection.insert_one(image.dict(exclude_unset = True , by_alias = True))
        image.id = record.inserted_id
        return image
    
    async def get_image(self,image_id : str) :

        result = await self.collection.find_one(
            {
                "_id" : ObjectId(image_id)
            }
        )

        if result is None :
            return None
        
        return Image(**result)
    
    async def delete_records_by_project_id(self, project_id : ObjectId ) :
        result = await self.collection.delete_many({
            "project_id" : project_id
        })

        return result.deleted_count