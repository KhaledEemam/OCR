from fastapi import APIRouter , Depends
from helpers import get_settings , Settings

base_router = APIRouter(
    prefix = '/api/v1' ,
    tags = ['api_v1']
)

@base_router.get('/')
async def welcome(app_settings : Settings = Depends(get_settings)) :

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {
        "APP name" : app_name ,
        "APP Version" : app_version
    }