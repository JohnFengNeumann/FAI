import torch
from loguru import logger


class FAIPipeline:
    def __init__(
        self,
        extractor,
        dataloader,
        bunch_size_seconds=0,
        bunch_size=16000,
        init_batch_size=1,
        batch_size_mode="auto",
    ):
        """
        batch_size_mode: 'auto' or 'manual', if 'auto', the batch size will be adjusted automatically, otherwise, the batch size will be fixed to init_batch_size
        """
        self.extractor = extractor
        self.sample_rate = self.extractor.sample_rate
        self.bunch_size = (
            bunch_size
            if bunch_size_seconds == 0
            else bunch_size_seconds * self.sample_rate
        )
        self.batch_size = (
            init_batch_size
            if batch_size_mode == "manual"
            else self.get_max_batch_size()
        )
        self.dataloader = dataloader

    def create_temp_audio(self, batch_size):
        return torch.ones(batch_size, self.bunch_size)

    def get_max_batch_size(self):
        # 初始批量大小
        batch_size = 1
        max_batch_size = 1

        # GPU可用性检查
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 指数增长阶段
        while True:
            try:
                # 创建临时音频数据
                temp_audio = self.create_temp_audio(batch_size).to(device)

                # 提取特征（测试当前批量大小是否可行）
                _ = self.extractor.extract_feature(temp_audio)

                # 如果成功，增加批量大小
                max_batch_size = batch_size
                logger.info(f"Current batch size: {batch_size}")
                batch_size *= 2  # 增长因子
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 如果发生 OOM，退出指数增长阶段
                    torch.cuda.empty_cache()
                    logger.warning(f"OOM occurs, current batch size: {batch_size}")
                    break
                else:
                    raise e

        # 使用二分查找进行精确搜索
        lower_bound = max_batch_size
        upper_bound = batch_size - 1  # 上限是发生OOM前的最大值

        while lower_bound <= upper_bound:
            mid_batch_size = (lower_bound + upper_bound) // 2
            try:
                # 创建临时音频数据
                temp_audio = self.create_temp_audio(mid_batch_size).to(device)

                # 测试当前批量大小
                _ = self.extractor.extract_feature(temp_audio)

                # 如果成功，更新最大值，并尝试更大的批量
                max_batch_size = mid_batch_size
                lower_bound = mid_batch_size + 1
                logger.info(
                    f"Batch size {mid_batch_size} succeeded, trying larger size..."
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 如果发生 OOM，尝试更小的批量
                    torch.cuda.empty_cache()
                    upper_bound = mid_batch_size - 1
                    logger.warning(
                        f"OOM occurs at batch size {mid_batch_size}, trying smaller size..."
                    )
                else:
                    raise e

        logger.info(f"Maximum batch size determined: {max_batch_size}")
        return max_batch_size
    
    def extract_feature(self):
        for batch_data in self.dataloader:
            audio_data = batch_data['audio_data']
            features = self.extractor.extract_feature(audio_data)
            
            data_info = batch_data['data_info']
            for d_info in data_info:
                start, end = d_info['start'], d_info['end']
                save_path = d_info['save_path']
                feature = features[start:end].unsqueeze(0)
                torch.save(feature, save_path)
            
