from models.enums import DataBaseEnum
from models.db_schemas import Asset
from .BaseDataModel import BaseDataModel
from bson.objectid import ObjectId

class AssetDataModel(BaseDataModel) :

    def __init__(self,db_client: object ) :
        super().__init__(db_client=db_client)
        self.collection = db_client[DataBaseEnum.ASSETS_COLLETIONS.value]

    @classmethod
    async def create_instance(cls , db_client : object) :
        instance = cls(db_client)
        await instance.init_collection()
        return instance
    
    async def init_collection(self) :
        all_collections = await self.db_client.list_collection_names()
        if self.collection not in all_collections :
            self.collection = self.db_client[DataBaseEnum.ASSETS_COLLETIONS.value]
            indexes = Asset.get_indexes()
            for index in indexes :
                await self.collection.create_index(
                    keys=index["keys"] ,
                    name=index["name"] ,
                    unique=index["unique"]
                )

    async def create_asset(self, asset : Asset) :
        record = await self.collection.insert_one(asset.dict(exclude_unset = True , by_alias = True))
        asset.id = record.inserted_id
        return asset
    
    async def delete_records_by_project_id(self, project_id : ObjectId ) :
        result = await self.collection.delete_many({
            "asset_project_id" : project_id
        })

        return result.deleted_count