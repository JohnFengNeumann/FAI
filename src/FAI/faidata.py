from loguru import logger
import os
import pickle
import multiprocessing
from glob import glob
import itertools
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import soundfile as sf
import math
import torchaudio as ta
from torch.utils.data import Sampler

# For multiprocessing
def get_audio_info(params):
    audio_path, chunk_size, model_sample_rate = params
    audio_length = len(sf.read(audio_path)[0])  # 音频样本总数
    audio_sample_rate = sf.info(audio_path).samplerate  # 音频采样率
    duration = audio_length / audio_sample_rate  # 音频时长
    # 计算每个 chunk_size 的持续时间，并向上取整
    duration_per_chunksize = math.ceil(
        audio_length / chunk_size if model_sample_rate == audio_sample_rate
        else duration * model_sample_rate / chunk_size
    )
    return audio_path, duration, duration_per_chunksize




class FAIDataset(Dataset):
    def __init__(self, data_path, dataset_type, save_path, read_database_procs = 2, database_path="database.csv", chunk_size_seconds = 0, chunk_size=131584, sample_rate=16000):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.database_path = database_path
        self.save_path = save_path
        self.read_database_procs = read_database_procs
        self.sample_rate = sample_rate
        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_size = chunk_size if chunk_size_seconds <= 0 else int(chunk_size_seconds * sample_rate)

        self.database = self.get_database() # database is a pandas dataframe with (audio_path, save_path, duration, duration_per_chunksize) columns

    def read_from_database_cache(self, audio_paths):
        # 初始化空的 DataFrame
        database = pd.DataFrame(columns=['audio_path', 'save_path', 'duration', 'duration_per_chunksize'])
        if os.path.isfile(self.database_path):  # 检查数据库文件是否存在
            logger.info(f"Found database cache file: {self.database_path}")
            old_database = pd.read_csv(self.database_path)
            logger.info(f"Loaded {len(old_database)} rows from database cache.")
        else:  # 数据库文件不存在时返回
            logger.info(f"Database cache file not found: {self.database_path}")
            return audio_paths, database

        # 创建一个集合存储需要排除的音频路径
        audio_paths_set = set(audio_paths)
        rows_to_keep = []

        # 筛选数据库中已存在的音频路径
        for old_audio_path in old_database['audio_path']:
            if old_audio_path in audio_paths_set:
                rows_to_keep.append(old_database[old_database['audio_path'] == old_audio_path])
                audio_paths_set.remove(old_audio_path)

        # 转换筛选后的行并合并为 DataFrame
        if rows_to_keep:
            database = pd.concat(rows_to_keep, ignore_index=True)
            logger.info(f"Old database was used for {len(database)} audios.")

        # 返回未找到的音频路径和更新后的数据库
        audio_paths = list(audio_paths_set)
        return audio_paths, database

    def get_database(self):
        read_database_procs = multiprocessing.cpu_count() if self.read_database_procs <= 0 else self.read_database_procs
        assert read_database_procs > 0, "read_database_procs must be greater than 0."

        logger.info(f"Dataset type: {self.dataset_type}, Processes to use: {read_database_procs}")
        logger.info(f"Collecting database for {self.data_path}")
        
        if self.dataset_type == 1:  # dataset_type 1 is a folder of audio files
            audio_paths = []
            if isinstance(self.data_path, list):# If multiple data paths are provided
                for tp in self.data_path:
                    audios_for_folder = sorted(glob(tp + "/*"))
                    if not audios_for_folder:
                        logger.warning(f"Warning: no audios found in folder '{tp}'. Please check it!")
                    audio_paths += audios_for_folder
            else:
                audio_paths += sorted(glob(self.data_path + "/*"))

            # only the wav, flac, mp3 files are supported
            audio_paths = [path for path in audio_paths if path.endswith('.wav') or path.endswith('.flac') or path.endswith('.mp3')]
            audio_paths, database = self.read_from_database_cache(audio_paths)
            logger.info(f"There are {len(audio_paths)} audios to process.")
            
            if read_database_procs <= 1:
                for path in tqdm(audio_paths):
                    audio_path, duration, duration_per_chunksize = get_audio_info((path, self.chunk_size, self.sample_rate))
                    new_row = pd.DataFrame([{
                        'audio_path': audio_path,
                        'save_path': self.save_path,
                        'duration': duration,
                        'duration_per_chunksize': duration_per_chunksize
                    }])
                    database = pd.concat([database, new_row], ignore_index=True)
            else:
                p = multiprocessing.Pool(processes=read_database_procs)
                with tqdm(total=len(audio_paths)) as pbar:
                    audio_info_iter = p.imap(
                        get_audio_info,
                        zip(audio_paths, itertools.repeat(self.chunk_size), itertools.repeat(self.sample_rate)),
                    )
                    new_rows = []
                    for audio_path, duration, duration_per_chunksize in audio_info_iter:
                        new_rows.append({
                            'audio_path': audio_path,
                            'save_path': self.save_path,
                            'duration': duration,
                            'duration_per_chunksize': duration_per_chunksize
                        })
                        pbar.update()
                    new_rows_df = pd.DataFrame(new_rows)
                    database = pd.concat([database, new_rows_df], ignore_index=True)
                p.close()

        # Save database
        database.to_csv(self.database_path, index=False)
        logger.info(f"Database saved to {self.database_path}, {len(database)} rows.")
        return database

    def __len__(self):
        return len(self.database)
    
    def __getitem__(self, idx):
        audio_path = self.database.iloc[idx]['audio_path']
        save_path = self.database.iloc[idx]['save_path']
        duration = self.database.iloc[idx]['duration']
        duration_per_chunksize = self.database.iloc[idx]['duration_per_chunksize']
        audio, sample_rate = ta.load(audio_path)
        # stereo to mono
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            audio = ta.transforms.Resample(sample_rate, self.sample_rate)(audio)
        return audio_path, save_path, duration, duration_per_chunksize, audio
    
    def collact_fn(self, batch):
        audio_data = []
        data_info = []
        start = 0
        end = 0
        for audio_path, save_path, duration, duration_per_chunksize, audio in batch:
            for i in range(0, audio.shape[1], self.chunk_size):
                chunk_audio = audio[:, i:i+self.chunk_size]
                if chunk_audio.shape[1] < self.chunk_size:
                    chunk_audio = ta.transforms.PadTrim(self.chunk_size)(chunk_audio)
                audio_data.append(chunk_audio)
            start = end
            end += duration_per_chunksize
            data_info.append({
                'start': start,
                'end': end,
                'save_path': save_path
            })
            
        audio_data = ta.transforms.Stack()(audio_data)
        return {'audio_data': audio_data, 'data_info': data_info}
            

class FAISampler(Sampler):
    def __init__(self, data_source: FAIDataset, batch_size):
        self.data_source = data_source
        self.data_base = data_source.database.copy()
        self.batch_size = batch_size
        self.chunk_size = self.data_source.chunk_size
        
    def __iter__(self):
        # 对数据库副本进行排序，以便优先选取较大的 duration_per_chunksize
        self.data_base = self.data_base.sort_values(by="duration_per_chunksize", ascending=False).reset_index(drop=True)
        indices = []  # 存储所有选取的音频索引
        current_batch = []  # 当前 batch 中的音频索引
        current_duration = 0  # 当前 batch 的总 duration
        
        # 遍历数据，构造批次
        for idx, row in self.data_base.iterrows():
            duration = row['duration_per_chunksize']
            if current_duration + duration <= self.batch_size:
                # 如果加入当前音频不会超出 batch_size，加入当前 batch
                current_batch.append(idx)
                current_duration += duration
            else:
                # 如果超出 batch_size，则结束当前 batch，并开始新的 batch
                if current_batch:  # 确保 batch 不为空
                    indices.append(current_batch)
                current_batch = [idx]
                current_duration = duration
            
            # 删除已处理的音频
            self.data_base.drop(idx, inplace=True)
        
        # 处理最后一个 batch（如果有剩余）
        if current_batch:
            indices.append(current_batch)
        
        # Flatten 所有 batch 的索引列表
        flat_indices = [item for batch in indices for item in batch]
        return iter(flat_indices)
        

            