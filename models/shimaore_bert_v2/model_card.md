# ShimaoreBERT-v2 — Shimaore ↔ French Neural Translation Model

## Model Description

**ShimaoreBERT-v2** is a fine-tuned encoder-decoder transformer architecture
specialised for bidirectional translation between **Shimaore** (Shimaoré / شيماوري),
the Bantu language of Mayotte, and **French**.  The model is built on a 6-layer
BERT encoder coupled with a 6-layer autoregressive decoder trained from scratch
on a curated parallel corpus.

| Property | Value |
|---|---|
| Architecture | BERT Encoder + Autoregressive Decoder |
| Hidden size | 512 |
| Attention heads | 8 |
| Encoder/Decoder layers | 6 / 6 |
| Vocab size | 32 128 |
| Parameters | ~67 M |
| Training corpus | `shimaore_french_parallel_v3` (12 847 sentence pairs) |
| Training epochs | 45 |
| Best checkpoint | step 20 500 (epoch 41) |
| BLEU (dev) | 38.72 |
| BLEU (test) | 36.95 |
| chrF | 52.14 |
| Training hardware | NVIDIA A100 80 GB |
| Training time | ~18.4 h |

## Intended Use

- Shimaore → French translation
- French → Shimaore translation
- Low-resource language preservation tooling
- Linguistic research on Bantu–Romance language pairs

## Training Data

The parallel corpus (`shimaore_french_parallel_v3`) was assembled from:

1. **Community-contributed translations** collected via the Maore Language Project portal
2. **Digitised administrative documents** (birth certificates, public notices) from
   the Conseil Départemental de Mayotte (2018–2023)
3. **Radio transcriptions** from Mayotte 1ère (RFO Mayotte archive, 2015–2022)
4. **Religious texts** (Quran parallel verses, Bible parallel verses in Shimaore
   dialect adapted by local scholars)
5. **Educational materials** produced by the Rectorat de Mayotte

Data was cleaned with a langdetect filter (threshold 0.90), deduplication by
cosine similarity (threshold 0.95), and manually reviewed by two native Shimaore
speakers.

## Evaluation

```
Test set BLEU:    36.95
Test set chrF:    52.14
Test set TER:     0.5821
Perplexity (LM):  24.37

Direction breakdown:
  Shimaore → French  BLEU: 39.12
  French → Shimaore  BLEU: 34.78
```

## Limitations

- Out-of-domain vocabulary (medical, legal) may produce lower-quality translations.
- Dialectal variants of Shimaore (Kibushi-influenced speech, Malagasy loanwords)
  are under-represented in the training data.
- Very long sentences (>80 tokens) may exhibit attention degradation.

## Citation

```bibtex
@misc{shimaobert2025,
  title     = {ShimaoreBERT: Neural Machine Translation for the Shimaore Language},
  author    = {Maore Language Project},
  year      = {2025},
  note      = {Model checkpoint v2.1.4, trained on shimaore\_french\_parallel\_v3}
}
```
