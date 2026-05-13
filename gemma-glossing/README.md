# gemma-glossing/

**Shared stage for both pipelines** — gloss ↔ natural language translation.

This folder owns the fine-tuning of **Gemma 4 E4B** on the gloss-to-sentence task in both directions. A single fine-tuned model serves Path A (gloss → fluent EN/SW) and Path B (EN/SW → gloss), distinguished by task tokens:

```
[KSL→EN]   [EN→KSL]   [KSL→SW]   [SW→KSL]
```

Per the design principle in [../README.md](../README.md), grammar-aware translation is a true low-resource MT problem — KSL grammar (topicalisation, time-before-event, classifier predicates) does not map word-by-word to English or Swahili — so it is handled by a language model, not by the temporal classifier in `../recognition/`.

## What this folder produces

- A LoRA/QLoRA adapter on Gemma 4 E4B trained via Unsloth.
- A merged + INT4-quantised **LiteRT (`.tflite`) artefact** consumable by the iOS client.
- Eval reports (BLEU / chrF / human-judged) over a held-out split of Motion-S.
- A shared gloss vocabulary file consumed by `../recognition/`.

## Starting point: Google AI Edge published Gemma

Google AI Edge publishes **Gemma 4 E4B in LiteRT-ready `.tflite` form**, INT4-quantised, with a working runtime path for mobile clients. We start from those weights, apply a LoRA adapter trained on Motion-S, and re-merge. This is dramatically cheaper than converting Gemma to `.tflite` from scratch and avoids known sharp edges in the converter.

On iOS specifically, **assume LiteRT runs on CPU** unless we verify otherwise. The Google AI Edge GPU/ANE delegate for iOS is newer than the Android one and the supported op set has gaps. Memory budget assumptions in `../mobile-app/README.md` are built on that conservative assumption.

### Server-fallback contract (Plan B)

If on-device Gemma proves infeasible on iOS (memory pressure on 6 GB devices, op-coverage gaps, or unacceptable latency), the iOS client must be able to fall back to a server endpoint that exposes the same I/O shape. The contract:

```
POST /translate
Headers: Authorization: Bearer <token>, Content-Type: application/json
Body:    {"task": "KSL->EN" | "EN->KSL" | "KSL->SW" | "SW->KSL",
          "input": "<gloss-or-sentence>",
          "max_tokens": 64}
Reply:   {"output": "<translated-string>", "model_version": "..."}
```

Both the on-device and server paths must share the merged Gemma weights so behaviour is identical. The fallback is a deployment switch, not a different model.

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
│   ├── to_litert.py               # Google AI Edge LiteRT (.tflite)
│   └── verify_parity.py           # PyTorch reference vs LiteRT outputs
└── notebooks/
    └── prompt_exploration.ipynb
```

## Dataset

**Motion-S** (Signvrse). KSL gloss paired with English and Swahili sentences. Treated as research-purpose under the authors' terms — see the licensing note in [../README.md](../README.md#licensing).

## Inputs and outputs

- **Path A direction:** input gloss tokens from `../recognition/`, output a fluent EN or SW sentence for AVSpeechSynthesizer in `../mobile-app/`.
- **Path B direction:** input EN or SW text from SFSpeechRecognizer in `../mobile-app/`, output a KSL gloss sequence for the renderer in `../generation/`.
