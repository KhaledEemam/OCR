from fastapi import APIRouter , UploadFile , Depends , status , Request , Response
from fastapi.responses import JSONResponse
from helpers import Settings , get_settings
from controllers import DataController , ProjectController
import os
import aiofiles
import logging
from models import ResponseSignal
from models import ProjectDataModel , ImagesDataModel , AssetDataModel
from datetime import datetime
from models.db_schemas import Image , Asset
from models.enums import AssetEnum
from tasks import get_predictions
import cv2
import base64

logger = logging.getLogger('uvicorn.error')
data_router = APIRouter(
    prefix = "/api/v1/data" ,
    tags = ['api_v1','data']
)


@data_router.get('/process/{project_id}')
async def upload_file(request : Request , project_id : str , file : UploadFile ,
                      app_settings : Settings=Depends(get_settings) , do_reset : int = 0 ) :
    
    db_client = request.app.db_client
    project_model = await ProjectDataModel.create_instance( db_client = db_client)
    image_model = await ImagesDataModel.create_instance( db_client = db_client)
    asset_model = await AssetDataModel.create_instance( db_client = db_client)
    project = await project_model.get_or_create_project(project_id=project_id)

    data_controller = DataController()
    is_valid , response_message = data_controller.validate_uploaded_data(file=file)

    if not is_valid :
        return JSONResponse(
            status_code = status.HTTP_400_BAD_REQUEST ,
            content = {
                'signal' : response_message
            }
        )

  

    file_path , file_id = data_controller.get_unique_file_path( project_id = project_id , original_file_name = file.filename )
    
    try :

        async with aiofiles.open(file_path , 'wb') as f :
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE) :
                await f.write(chunk) 

    except Exception as e :

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse (
            status_code = status.HTTP_400_BAD_REQUEST ,
            content = ResponseSignal.FILE_UPLOAD_FAILED.value
        )
    
    image = Image(
        image_path = file_path ,
        project_id = project.id
        )
    
    asset = Asset (
        asset_project_id = project.id ,
        asset_type = AssetEnum.FILE.value , 
        asset_name = file_id  , 
        asset_size = os.path.getsize(file_path)
    )
    
    if do_reset : 
        await image_model.delete_records_by_project_id( project_id = project.id)
        await asset_model.delete_records_by_project_id( project_id = project.id)

    image = await image_model.insert_image(image)
    asset = await asset_model.create_asset(asset=asset)

    output_image ,text_from_image = get_predictions(image_path= file_path , nms_threshold = .1 ,certainty_score = .9)
    
    # Encode image as JPEG
    _, encoded_image = cv2.imencode('.jpg', output_image)

    # Convert the encoded image to bytes
    image_bytes = encoded_image.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    return JSONResponse(
        content = {"signal" : response_message ,"text" : text_from_image , "aaa" : image_base64 },
        )