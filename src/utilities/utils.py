import psutil
import multiprocessing as mp

DEFAULT_DATASET_NAME: str = "trec-covid"
AVAILABLE_CORES: int = mp.cpu_count()
AVAILABLE_RAM_GB: int = psutil.virtual_memory().total // 10**9
