import os


def pytest_configure(config):
    # 在 pytest 启动时设置 CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第 0 和第 1 块 GPU
    ####################################################################################################

    from pathlib import Path
    import sys

    root_dir = Path(__file__).parent.parent
    sys.path.append(str(root_dir))
    ####################################################################################################
