from pydantic import BaseSettings

class AppConfig(BaseSettings):
    num_columns: int = 6

    thumbnail_size: tuple = (250, 250)

    triton_url: str = "http://localhost:8000"
    model_name: str = "litehrnet"
    model_version: str = "1"
    use_http: bool = True

    class Config:
        env_prefix = 'APP_'


config = AppConfig()