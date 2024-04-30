from pathlib import Path
import os
from loguru import logger


def index(model_name_or_path: str = None, dataset_name_or_path: str = None):
    model_name_or_path = (
        "intfloat/multilingual-e5-base"
        if model_name_or_path in ("None", None)
        else model_name_or_path
    )
    dataset_name_or_path = (
        "samples" if dataset_name_or_path in ("None", None) else dataset_name_or_path
    )
    logger.info(
        f"INDEX pipeline using model=[{str(model_name_or_path)}] and dataset=[{dataset_name_or_path}]"
    )
