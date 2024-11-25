import torch
import numpy as np
from fairseq import checkpoint_utils


class ContentVec768L12:
    def __init__(
        self,
        vec_path="ckpt/contentvec/checkpoint_best_legacy_500.pt",
        device=None,
    ):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 768
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [vec_path],
            suffix="",
        )
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = models[0].to(self.device)
        self.model.eval()
        self.sample_rate = 16000

    def extract_feature(self, wav, return_numpy=False):
        if isinstance(wav, np.ndarray):
            wav = torch.FloatTensor(wav)
        feats = wav.view(1, -1) if len(wav.shape) == 1 else wav  # (batch_size, seq_len)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask.to(self.device),
            "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)

        if return_numpy:
            return logits[0].transpose(1, 2).cpu().numpy()
        else:
            return logits[0].transpose(1, 2).cpu()
