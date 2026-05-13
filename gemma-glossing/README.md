# gemma-glossing/

**Shared stage for both pipelines** — gloss ↔ natural language translation.

This folder owns the fine-tuning of **Gemma 4 E4B** on the gloss-to-sentence task in both directions. A single fine-tuned model serves Path A (gloss → fluent EN/SW) and Path B (EN/SW → gloss), distinguished by task tokens:

```
[KSL→EN]   [EN→KSL]   [KSL→SW]   [SW→KSL]
```

Per the design principle in [../README.md](../README.md), grammar-aware translation is a true low-resource MT problem — KSL grammar (topicalisation, time-before-event, classifier predicates) does not map word-by-word to English or Swahili — so it is handled by a language model, not by the temporal classifier in `../recognition/`.

## What this folder produces

- A LoRA/QLoRA adapter on Gemma 4 E4B trained via Unsloth.
- A merged + INT4-quantised LiteRT export for on-device use on Android.
- Eval reports (BLEU / chrF / human-judged) over a held-out split of Motion-S.
- A shared gloss vocabulary file consumed by `../recognition/`.

## Intended layout

```
gemma-glossing/
├── README.md
├── requirements.txt
├── configs/
│   ├── unsloth_qlora.yaml         # primary fine-tune config
│   └── eval.yaml
├── data/
│   ├── load_motion_s.py           # ingest Signvrse Motion-S
│   ├── pair_builder.py            # build (gloss, EN, SW) tuples
│   ├── task_tokens.py             # prepend [KSL→EN] etc. for bidirectional
│   ├── splits/                    # train/val/test split manifests
│   └── vocab/
│       └── gloss_vocab.json       # shared with ../recognition/
├── prompts/
│   ├── ksl_to_en.txt
│   ├── en_to_ksl.txt
│   ├── ksl_to_sw.txt
│   └── sw_to_ksl.txt
├── training/
│   ├── finetune_unsloth.py
│   └── merge_lora.py
├── eval/
│   ├── run_eval.py
│   ├── metrics.py                 # BLEU, chrF, exact-gloss accuracy
│   └── error_analysis.ipynb
├── export/
│   ├── quantize_int4.py
│   ├── to_litert.py               # Google AI Edge LiteRT
│   └── verify_parity.py           # CPU reference vs LiteRT outputs
└── notebooks/
    └── prompt_exploration.ipynb
```

## Dataset

**Motion-S** (Signvrse). KSL gloss paired with English and Swahili sentences. Treated as research-purpose under the authors' terms — see the licensing note in [../README.md](../README.md#licensing).

## Inputs and outputs

- **Path A direction:** input gloss tokens from `../recognition/`, output a fluent EN or SW sentence for the Android TTS in `../mobile-app/`.
- **Path B direction:** input EN or SW text from the ASR in `../mobile-app/`, output a KSL gloss sequence for the renderer in `../generation/`.
