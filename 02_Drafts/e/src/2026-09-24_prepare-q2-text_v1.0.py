"""Cache frozen local BERT features for train/valid; never fit on held-out data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '03_Results/e/question-two/bert-cache-v1.0'
MODEL = Path(r'D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594')


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(str(a.shape).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def model_fingerprint(folder):
    weights = folder / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'Expected fixed local safetensors weights: {weights}')
    return {'snapshot': folder.name, 'path': str(folder),
            'files': {name: sha(folder / name) for name in ('config.json', 'model.safetensors')}}


def input_contract(batch, model_info):
    return {'split': batch.split, 'samples': len(batch), 'shape': [len(batch), 50, 768],
            'source_sha256': dict(batch.source_sha256),
            'ids_sha256': hashlib.sha256(json.dumps(batch.ids, ensure_ascii=False).encode('utf-8')).hexdigest(),
            'token_sha256': {name: array_sha(getattr(batch, name))
                             for name in ('input_ids', 'attention_mask', 'token_type_ids')},
            'model': model_info, 'representation': 'last_hidden_state', 'dtype': 'float32',
            'model_mode': 'eval_frozen', 'special_tokens': 'included_for_encoding_excluded_from_content_pooling',
            'padding': 'embeddings_retained_use_loader_content_mask'}


def verify_cache(batch, model_info, out=OUT):
    path = out / f'{batch.split}.npy'
    meta_path = out / f'{batch.split}.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta['contract'] != input_contract(batch, model_info):
        raise RuntimeError(f'{batch.split}: cache source/model/token contract changed')
    if sha(path) != meta['feature_sha256']:
        raise RuntimeError(f'{batch.split}: cache hash mismatch')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.shape != (len(batch), 50, 768) or values.dtype != np.float32:
        raise RuntimeError(f'{batch.split}: invalid feature shape/dtype')
    for start in range(0, len(batch), 64):
        if not np.isfinite(values[start:start+64]).all():
            raise RuntimeError(f'{batch.split}: nonfinite cached values')
    return meta


def render_report(metadata):
    lines = ['# 统一 BERT 文本编码缓存', '',
             '固定本地 `google-bert/bert-base-uncased` 快照，将官方 `text_bert` 直接输入冻结的 BERT，保存最后一层 float32 向量。原文件只读，未训练情感模型。本轮仅编码 train / valid；test 与附件三、四未运行编码或预测。', '',
             '所有位置（含特殊 token 和 padding）保留原索引；建模时必须使用加载器的内容/观测掩码。音频与视觉仍读取官方特征。下表差值仅比较内容位置与原包的预计算 `text`，用于追溯数值兼容性。', '',
             '| 划分 | 样本 | 输出形状 | 对照最大绝对差 | 对照平均绝对差 |',
             '|---|---:|---|---:|---:|']
    for key, meta in metadata.items():
        ref = meta['reference_comparison']
        lines.append(f"| {key} | {meta['contract']['samples']} | {meta['contract']['shape']} | {ref['max_absolute_error']:.8g} | {ref['mean_absolute_error']:.8g} |")
    lines += ['', '缓存带原 PKL、token、样本顺序、模型配置和权重的 SHA-256；每份 NPY 有独立文件指纹。读取检查会拒绝内容、模型或顺序变化后的旧缓存。', '',
              '没有重新分词、改写原文或用 ASR 替换官方 token。专项集合后续必须调用同一编码函数与模型快照，不得仅因附件三没有 `text` 就填零。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify source, model, tokens and saved arrays without model inference or writing')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    loader = module('q2_data_text_cache', '2026-09-24_q2_data_v1.0.py')
    data = loader.load_official()
    model_info = model_fingerprint(MODEL)
    metadata = {}
    encoder = torch = transformers = None
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid'):
        batch = data[split]
        path = OUT / f'{split}.npy'
        meta_path = OUT / f'{split}.json'
        if args.check or (path.exists() and meta_path.exists()):
            metadata[split] = verify_cache(batch, model_info)
            print(f'{split}: verified cached {len(batch)} samples', flush=True)
            continue
        if path.exists() or meta_path.exists():
            raise RuntimeError(f'{split}: incomplete committed cache; investigate before replacing')
        if encoder is None:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            import torch
            import transformers
            from transformers import BertModel
            if args.device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but not available')
            torch.manual_seed(20260924)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            encoder = BertModel.from_pretrained(str(MODEL), local_files_only=True,
                                                attn_implementation='eager').to(args.device).float().eval()
            encoder.requires_grad_(False)
            print(f'Frozen local BERT ready: {args.device}, float32, eager attention', flush=True)
        partial = path.with_suffix('.partial.npy')
        values = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                           shape=(len(batch), 50, 768))
        max_error = total_error = 0.0
        compared = 0
        with torch.inference_mode():
            for start in range(0, len(batch), args.batch_size):
                end = min(start + args.batch_size, len(batch))
                inputs = {key: torch.tensor(getattr(batch, key)[start:end], dtype=torch.long, device=args.device)
                          for key in ('input_ids', 'attention_mask', 'token_type_ids')}
                result = encoder(**inputs).last_hidden_state.cpu().numpy()
                if result.shape != (end-start, 50, 768) or not np.isfinite(result).all():
                    raise RuntimeError(f'{split}: invalid BERT output at {start}')
                values[start:end] = result
                active = batch.content_mask[start:end]
                delta = np.abs(result[active].astype(np.float64) - batch.reference_text[start:end][active])
                max_error = max(max_error, float(delta.max()))
                total_error += float(delta.sum())
                compared += delta.size
                if start == 0 or end % (args.batch_size * 25) == 0 or end == len(batch):
                    print(f'{split}: encoded {end}/{len(batch)}', flush=True)
        values.flush()
        del values
        meta = {'contract': input_contract(batch, model_info), 'feature_sha256': sha(partial),
                'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                            'device': args.device, 'batch_size': args.batch_size, 'seed': 20260924,
                            'attention_implementation': 'eager', 'allow_tf32': False},
                'reference_comparison': {'positions': int(batch.content_mask.sum()),
                                         'max_absolute_error': max_error,
                                         'mean_absolute_error': total_error / compared}}
        partial.replace(path)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        metadata[split] = verify_cache(batch, model_info)
    report = OUT.parent / '2026-09-24_q2-text-cache_v1.0.md'
    rendered = render_report(metadata)
    if args.check:
        if report.read_text(encoding='utf-8') != rendered:
            raise RuntimeError('Text cache report mismatch')
    else:
        report.write_text(rendered, encoding='utf-8')
    print('Train/valid text cache verified; no task training or held-out inference', flush=True)


if __name__ == '__main__':
    main()
