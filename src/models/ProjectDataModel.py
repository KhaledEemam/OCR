from models.enums import DataBaseEnum
from models.db_schemas import Project
from .BaseDataModel import BaseDataModel
from bson.objectid import ObjectId

class ProjectDataModel(BaseDataModel) :
    def __init__(self , db_client : object) :
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.PROJECTS_COLLECTION.value]

    @classmethod
    async def create_instance(cls , db_client : object) :
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self) :
        all_collections = await self.db_client.list_collection_names()
        if self.collection not in all_collections :
            self.collection = self.db_client[DataBaseEnum.PROJECTS_COLLECTION.value]
            indexes = Project.get_indexes()
            for index in indexes :
                await self.collection.create_index(
                    keys=index["keys"] ,
                    name=index["name"] ,
                    unique=index["unique"]
                )


    async def create_project(self, project : Project) :

        result = await self.collection.insert_one(project.dict(by_alias=True , exclude_unset = True))
        project.id = result.inserted_id

        return project
    
    async def get_or_create_project(self, project_id : str) :

        record = await self.collection.find_one(
            {
                "project_id" : project_id
            }
        )

        if record is None :

            project = Project(project_id=project_id)
            project = await self.create_project(project=project)

            return project

        return Project(**record)
    
    async def get_all_projects(self, page : int = 1 , page_size : int = 10) :

        total_documents = await self.collection.count_documents({})

        no_of_pages = total_documents // page_size

        if total_documents % page_size > 0 :
            no_of_pages += 1

        cursor = await self.collection.find().skip( (page-1) * page_size ).limit(page_size)
        projects = []
        
        async for document in cursor :
            projects.append(
                Project(**document)
            )

        return projects , no_of_pages