import os
import pytest

from src.FAI import FAIPipeline
from modules.contentvec import ContentVec768L12
from loguru import logger


def test_faipipeline_init_second2():
    contentvec_extractor = ContentVec768L12()
    pipeline = FAIPipeline(contentvec_extractor, bunch_size_seconds=2)

    assert pipeline.sample_rate == 16000
    assert pipeline.bunch_size == 32000
    logger.info(f"pipeline.batch_size: {pipeline.batch_size}")


def test_faipipeline_init_second5():
    contentvec_extractor = ContentVec768L12()
    pipeline = FAIPipeline(contentvec_extractor, bunch_size_seconds=5)

    assert pipeline.sample_rate == 16000
    assert pipeline.bunch_size == 80000
    logger.info(f"pipeline.batch_size: {pipeline.batch_size}")
