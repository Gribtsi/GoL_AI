from config import INFERENCE_SERVICES_COUNT

DATASETS_FILEPATH = "/data/dataset.h5"
WRITER_ADDRESS = "tcp://data_writer:5555"

MODEL_PROVIDER_ADDRESS = "tcp://model_provider:5556"


MODELS_DIR = "/models"

DATASET_COUNT_ADDRESS = "tcp://data_writer:5557"

def get_inference_service_address(process_id : int):

    if process_id < 0:
        process_id = 0

    return f"tcp://inference_server:{5600 + process_id % INFERENCE_SERVICES_COUNT}"