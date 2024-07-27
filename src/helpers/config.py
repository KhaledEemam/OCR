from pydantic_settings import BaseSettings , SettingsConfigDict

class Settings(BaseSettings) :
    AANOTATIONS_PATH : str
    IMAGES_INFO_PATH : str
    IMAGES_PATH : str
    PROCESSED_DETECTOR_DATA : str
    SAVED_DETECTOR_MODEL_PATH : str
    SAVED_TEXT_RECOGNIZER_MODEL_PATH : str
    APP_NAME : str
    APP_VERSION : str
    ACCEPTED_FILE_TYPES : list
    MAX_INPUT_SIZE : int
    FILE_DEFAULT_CHUNK_SIZE : int
    MONGODB_URL : str
    MONGODB_DATABASE : str

    class Config :
        env_file = '.env'


def get_settings() :
    return Settings()