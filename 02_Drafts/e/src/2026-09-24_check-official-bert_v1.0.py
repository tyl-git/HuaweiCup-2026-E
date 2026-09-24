"""Compare one official text embedding against local BERT inference."""

from pathlib import Path
import json
import pickle

import numpy as np
import torch
from transformers import BertModel


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "01_Source" / "E" / "E题数据" / "附件2-数据集特征文件" / "aligned_50.pkl"
MODEL = Path(r"D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594")
OUTPUT = ROOT / "03_Results" / "e" / "question-two" / "2026-09-24_official-bert-comparison_v1.0.json"


def main() -> None:
    with SOURCE.open("rb") as handle:
        data = pickle.load(handle)["train"]
    model = BertModel.from_pretrained(str(MODEL), local_files_only=True).eval()
    inputs = data["text_bert"][0]
    active = inputs[1].astype(bool)
    old = data["text"][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.inference_mode():
        new = model(
            input_ids=torch.tensor(inputs[0:1], dtype=torch.long, device=device),
            attention_mask=torch.tensor(inputs[1:2], dtype=torch.long, device=device),
            token_type_ids=torch.tensor(inputs[2:3], dtype=torch.long, device=device),
        ).last_hidden_state[0].cpu().numpy()
    difference = np.abs(old - new)
    cosine = np.sum(old * new, axis=1) / (np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1))
    report = {
        "sample_id": data["id"][0],
        "model_snapshot": str(MODEL),
        "device": str(device),
        "active_tokens": int(active.sum()),
        "official_dtype": str(old.dtype),
        "computed_dtype": str(new.dtype),
        "max_absolute_difference_active": float(difference[active].max()),
        "mean_absolute_difference_active": float(difference[active].mean()),
        "cosine_active_min": float(cosine[active].min()),
        "cosine_active_mean": float(cosine[active].mean()),
        "max_absolute_difference_padding": float(difference[~active].max()),
        "mean_absolute_difference_padding": float(difference[~active].mean()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
