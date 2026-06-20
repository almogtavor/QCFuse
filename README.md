<table align="center">
  <tr>
    <td align="center" valign="middle" width="84">
      <img src="md/kv_fusion_icon_editable.svg" alt="QCFuse icon" width="68">
    </td>
    <td valign="middle">
      <h1>
        <strong>QCFuse</strong>: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving
      </h1>
    </td>
  </tr>
</table>

<br>

<p align="center">
  <a href="http://arxiv.org/abs/2606.05875"><img src="https://img.shields.io/badge/Paper-arXiv%3A2606.05875-b31b1b" alt="Paper"></a>
</p>

QCFuse is a **pipeline-constrained, query-aware KV cache fusion** system for
efficient long-context RAG generation. This repository contains the QCFuse
research artifact described in [arXiv:2606.05875](http://arxiv.org/abs/2606.05875).

## ✨ Highlights

<p align="center">
  <img src="md/3sys_framework_01.png" alt="QCFuse framework overview" width="95%">
</p>

<p align="center">
  <em>QCFuse builds a compact query-aware view for pipelined cache fusion in RAG serving.</em>
</p>

- **Query-aware compressed view.** Cuts selector time and selection noise while
  preserving pipeline efficiency.
- **Pipeline-aware SGLang system.** Adds SSD-backed PIC cache transfer and
  Triton sparse reconstruction attention without materializing extra attention
  masks.
- **Matched-quality speedup.** Reaches **1.7x** average prefill speedup over
  full prefill and **1.5x** over ProphetKV.
## 📊 Results

<p align="center">
  <img src="md/benchmark_aggregate.png" alt="Quality and TTFT trade-off on LongBench and RULER" width="55%">
</p>

<p align="center">
  <em>Quality and TTFT trade-off on LongBench and RULER. Transfer bandwidth affects the TTFT of KV cache fusion methods; paper experiments use 10GB/s.</em>
</p>

## 🗂️ Repository Layout

```text
QCFuse/
├── blend/                         # QCFuse evaluation runner and configs
│   ├── sglang_blend_ssd.py
│   ├── blend_common.py
│   ├── qcfuse_config.py
│   └── utils.py
├── srt/                           # SGLang runtime changes for QCFuse
│   ├── entrypoints/
│   ├── managers/
│   ├── layers/attention/
│   ├── models/
│   └── utils/
├── data/                          # Packaged evaluation data
│   └── qcfuse_data.zip
```

## 🗄️ Datasets

The evaluation data is provided as `data/qcfuse_data.zip`. It contains six
ready-to-run JSONL files derived from LongBench and RULER. Each dataset has
200 samples.

Extract the archive in the repository root:

```bash
unzip data/qcfuse_data.zip -d data
```

The evaluation runner expects each split as `{dataset}.jsonl` under
`--data_dir`; use `--data_dir data` after extraction. See
[data/README.md](data/README.md) for the packaged data layout.

| Benchmark | Official source | Tasks used in this artifact |
| --- | --- | --- |
| LongBench | [THUDM/LongBench](https://github.com/THUDM/LongBench) | `musique`, `2wikimqa`, `hotpotqa` |
| RULER | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | `ruler_mv` (`MV`), `ruler_mq` (`MQ`), `ruler_vt` (`VT`) |

## ⚙️ Installation

Download SGLang **0.5.4**, replace its `sglang/python/sglang/` package with
the QCFuse code, and then build it:

```bash
git clone -b v0.5.4 https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip
pip install -e "python"
```

Return to the QCFuse repository root before running the commands below.

Install the evaluation dependencies used by the Blend runner:

```bash
pip install rouge-score
```

Use a CUDA/PyTorch environment compatible with your GPU and SGLang 0.5.4. The
runner expects local model files and local JSONL datasets.

## 🚀 Running QCFuse

Run the SSD-backed QCFuse method:

```bash
python blend/sglang_blend_ssd.py \
  --model qwen3-8b \
  --model_dir models \
  --data_dir data \
  --dataset hotpotqa \
  --baseline ours \
  --size 200 \
  --cache_dir cache/qcfuse
```

`--cache_dir` stores the SSD-backed chunk and query caches. With
`--baseline ours`, the runner performs offline cache preparation before the
online evaluation pass. The default reconstruction ratio for `ours` is `0.5`.

Run the full-prefill baseline:

```bash
python blend/sglang_blend_ssd.py \
  --model qwen3-8b \
  --model_dir models \
  --data_dir data \
  --dataset hotpotqa \
  --baseline fullcomp \
  --size 200 \
  --cache_dir cache/qcfuse
```


## 📚 Citation

If you find QCFuse useful, please cite:

```bibtex
@misc{yan2026qcfusequeryawarecachefusion,
      title={QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving},
      author={Jianxin Yan and Wangze Ni and Zhenxin Li and Jiabao Jin and Zhitao Shen and Haoyang Li and Jia Zhu and Peng Cheng and Xuemin Lin and Lei Chen and Kui Ren},
      year={2026},
      eprint={2606.05875},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.05875},
}
```
