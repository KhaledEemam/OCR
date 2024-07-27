from fastapi import FastAPI
from routes import base_router , data_router
from motor.motor_asyncio import AsyncIOMotorClient
from helpers import get_settings


app = FastAPI()
app.include_router(base_router)
app.include_router(data_router)

@app.on_event("startup")
async def start_up_client() :
    settings = get_settings()
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]


@app.on_event("shutdown")
async def start_up_client() :
    settings = get_settings()
    app.mongo_conn.close()