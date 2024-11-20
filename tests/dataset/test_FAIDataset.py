####################################################################################################
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
####################################################################################################

import os
import pytest
import pandas as pd
import soundfile as sf
import numpy as np
from tempfile import TemporaryDirectory
from src.FAI import FAIDataset  # 替换为实际模块路径

# 创建测试用的音频文件
def create_test_audio(file_path, duration, sample_rate):
    samples = np.random.uniform(-1, 1, int(duration * sample_rate))
    sf.write(file_path, samples, sample_rate)

# 测试类的初始化
def test_faidataset_initialization():
    with TemporaryDirectory() as tmp_dir:
        dataset = FAIDataset(
            data_path=tmp_dir,
            dataset_type=1,
            save_path=os.path.join(tmp_dir, "save"),
            database_path=os.path.join(tmp_dir, "database.csv"),
            chunk_size_seconds=2,
            sample_rate=16000
        )
        assert dataset.data_path == tmp_dir
        assert dataset.sample_rate == 16000
        assert dataset.chunk_size == 32000  # 2 * 16000

# 测试数据库生成功能
def test_get_database():
    with TemporaryDirectory() as tmp_dir:
        # 创建虚拟音频文件
        audio_path_1 = os.path.join(tmp_dir, "test.wav")
        create_test_audio(audio_path_1, duration=10, sample_rate=16000)
        
        # 初始化数据集
        dataset = FAIDataset(
            data_path=tmp_dir,
            dataset_type=1,
            save_path=os.path.join(tmp_dir, "save"),
            database_path=os.path.join(tmp_dir, "database.csv"),
            chunk_size_seconds=2,
            sample_rate=16000
        )
        
        database = dataset.database
        assert len(database) == 1
        assert database.iloc[0]['audio_path'] == audio_path_1
        assert database.iloc[0]['duration'] == 10
        assert database.iloc[0]['duration_per_chunksize'] == 5  # 10s audio, chunk size = 2s
        
        # 检查是否保存到CSV
        assert os.path.isfile(dataset.database_path)
        saved_database = pd.read_csv(dataset.database_path)
        assert len(saved_database) == 1
        
        # 创建第二个音频文件,不同的sample_rate
        audio_path_2 = os.path.join(tmp_dir, "test2.wav")
        create_test_audio(audio_path_2, duration=5, sample_rate=32000)
        database = dataset.get_database()
        assert len(database) == 2
        assert database.iloc[0]['audio_path'] == audio_path_1
        assert database.iloc[0]['duration'] == 10
        assert database.iloc[0]['duration_per_chunksize'] == 5  # 10s audio, chunk size = 2s
        assert database.iloc[1]['audio_path'] == audio_path_2
        assert database.iloc[1]['duration'] == 5
        assert database.iloc[1]['duration_per_chunksize'] == 3

# 测试缓存读取功能
def test_read_from_database_cache():
    with TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, "test.wav")
        create_test_audio(audio_path, duration=10, sample_rate=16000)

        # 创建虚拟缓存
        database_path = os.path.join(tmp_dir, "database.csv")
        pd.DataFrame({
            'audio_path': [audio_path],
            'save_path': [os.path.join(tmp_dir, "save")],
            'duration': [10],
            'duration_per_chunksize': [5]
        }).to_csv(database_path, index=False)

        dataset = FAIDataset(
            data_path=tmp_dir,
            dataset_type=1,
            save_path=os.path.join(tmp_dir, "save"),
            database_path=database_path,
            chunk_size_seconds=2,
            sample_rate=16000
        )

        audio_paths, database = dataset.read_from_database_cache([audio_path])
        assert len(audio_paths) == 0  # 音频文件已在缓存中
        assert len(database) == 1
        assert database.iloc[0]['audio_path'] == audio_path

# 测试多进程功能
def test_multiprocessing_get_audio_info():
    with TemporaryDirectory() as tmp_dir:
        for audio_idx in range(10):
            audio_path = os.path.join(tmp_dir, f"test_{audio_idx}.wav")
            create_test_audio(audio_path, duration=10, sample_rate=16000)
    
        dataset = FAIDataset(
            data_path=tmp_dir,
            dataset_type=1,
            save_path=os.path.join(tmp_dir, "save"),
            database_path=os.path.join(tmp_dir, "database.csv"),
            chunk_size_seconds=2,
            sample_rate=16000
        )

        database = dataset.get_database()
        assert len(database) == 10
        assert all(database['duration'] == 10)


# 测试多文件夹
def test_multi_file_input():
    with TemporaryDirectory() as tmp_dir:
        os.mkdir(os.path.join(tmp_dir, 'test1'))
        os.mkdir(os.path.join(tmp_dir, 'test2'))
        for audio_idx in range(10):
            audio_path = os.path.join(tmp_dir, 'test1', f"test_{audio_idx}.wav")
            create_test_audio(audio_path, duration=10, sample_rate=16000)
        for audio_idx in range(10):
            audio_path = os.path.join(tmp_dir, 'test2', f"test_{audio_idx}.wav")
            create_test_audio(audio_path, duration=2, sample_rate=16000)
    
        dataset = FAIDataset(
            data_path=[os.path.join(tmp_dir, 'test1'), os.path.join(tmp_dir, 'test2')],
            dataset_type=1,
            save_path=os.path.join(tmp_dir, "save"),
            database_path=os.path.join(tmp_dir, "database.csv"),
            chunk_size_seconds=2,
            sample_rate=16000
        )

        database = dataset.get_database()
        assert len(database) == 20