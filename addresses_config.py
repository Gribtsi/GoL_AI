from config import INFERENCE_SERVICES_COUNT, PROCESSES_PER_WORKER

DATASETS_FILEPATH = "/data/dataset.h5"
WRITER_ADDRESS = "tcp://data_writer:5555"

TOURNAMENT_ACK_ADDRESS = "tcp://tournament_manager:5800"


MODEL_PROVIDER_ADDRESS = "tcp://model_provider:5556"

TOURNAMENT_DATASETS_FILEPATH  = "/data/tournament_games.hdf5"
TOURNAMENT_RESULTS_FILEPATH   = "/data/tournament_results.jsonl"

DATASETS_DIR = "/data"
MODELS_DIR = "/models"

DATASET_COUNT_ADDRESS = "tcp://data_writer:5557"

def get_inference_service_address(process_id : int):

    if process_id < 0:
        process_id = 0

    return f"tcp://inference_server:{5600 + process_id % INFERENCE_SERVICES_COUNT}"

WORKER_CONTROL_BASE_PORT = 5700
def get_worker_control_address(process_id: int) -> str:
    return f"tcp://worker:{WORKER_CONTROL_BASE_PORT + process_id % PROCESSES_PER_WORKER}"
