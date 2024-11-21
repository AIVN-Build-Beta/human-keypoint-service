from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    num_columns: int = 6

    thumbnail_size: tuple = (250, 250)

    triton_url: str = "localhost:8000"
    model_name: str = "lite_hrnet_model"
    model_version: str = "1"
    use_http: bool = True

    prometheus_port: int = 8005

    class Config:
        env_prefix = 'APP_'


app_config = AppConfig()