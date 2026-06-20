# Evaluation Data

This directory provides the packaged QCFuse evaluation data:

```text
data/qcfuse_data.zip
```

The archive contains six ready-to-run JSONL files:

- LongBench: `musique.jsonl`, `2wikimqa.jsonl`, `hotpotqa.jsonl`
- RULER: `ruler_mv.jsonl`, `ruler_mq.jsonl`, `ruler_vt.jsonl`

The files are derived from the official LongBench and RULER datasets. Each
dataset contains 200 samples. RULER samples are split into 20 chunks, while
LongBench samples are capped at 20 chunks and may contain fewer. The average
context length is about 10K tokens.

Extract the archive in the repository root:

```bash
unzip data/qcfuse_data.zip -d data
```

Use `--data_dir data` in the Blend runner after extraction.
