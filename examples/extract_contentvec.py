####################################################################################################
from pathlib import Path
import sys

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
####################################################################################################

import os
import torch

from src.FAI import FAIPipeline
from modules.contentvec import ContentVec768L12
from loguru import logger
from time import time


def faipipeline_init_second30():
    contentvec_extractor = ContentVec768L12()
    pipeline = FAIPipeline(contentvec_extractor, bunch_size_seconds=30)

    assert pipeline.sample_rate == 16000
    assert pipeline.bunch_size == 480000

    del contentvec_extractor
    del pipeline
    torch.cuda.empty_cache()


def faipipeline_init_second2():
    contentvec_extractor = ContentVec768L12()
    pipeline = FAIPipeline(contentvec_extractor, bunch_size_seconds=2)

    assert pipeline.sample_rate == 16000
    assert pipeline.bunch_size == 32000

    del contentvec_extractor
    del pipeline
    torch.cuda.empty_cache()


def faipipeline_init_second5():
    contentvec_extractor = ContentVec768L12()
    pipeline = FAIPipeline(contentvec_extractor, bunch_size_seconds=5)

    assert pipeline.sample_rate == 16000
    assert pipeline.bunch_size == 80000

    del contentvec_extractor
    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    time_start = time()
    faipipeline_init_second30()
    logger.info(f"Time elapsed: {time() - time_start:.6f}s")
    time_start = time()
    faipipeline_init_second2()
    logger.info(f"Time elapsed: {time() - time_start:.6f}s")
    time_start = time()
    faipipeline_init_second5()
    logger.info(f"Time elapsed: {time() - time_start:.6f}s")
