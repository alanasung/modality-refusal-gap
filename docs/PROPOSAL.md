# Proposal: Refusal Without a Projector

**Target project.** Locating the Refusal Circuit in Unified, Encoder-Free Vision-Language Models
**Mentor.** Alessandro Suglia; Rohit Saxena; Francesco Pinto (University of Edinburgh; University of Edinburgh; Google DeepMind)
**Research areas.** Mechanistic interpretability; AI security; Behavioral evaluation of LLMs
**Posting.** https://sparai.org/projects/f26/recTF6JBqSCgR6sNf

## Summary

If there is no vision projector to blame for the text-to-image safety gap, find out what is actually carrying it.

## Hypothesis

In a unified, encoder-free VLM the text-derived refusal direction still exists and is still linear, but image-delivered harmful intent lands with systematically lower projection onto it. The deficit accumulates across early decoder layers rather than appearing at one interface, which is the signature that distinguishes a distributed failure from a modular one.

A hypothesis worth testing has to be able to lose. This one loses if the
measurements below come back null, and the design is built so that a null is
reportable rather than a dead end.

## Research questions

1. How large is the text-to-image refusal gap on matched content, both for typographic renderings and for VLSBench leakless items where the harmful intent is genuinely visual and not recoverable from the text alone?
2. Does the refusal direction extracted from text transfer to the visual channel, measured by projection of image-prompt activations onto it?
3. Where does the deficit accumulate, layer by layer, and does patching text-derived activations into the image run restore refusal?
4. What does any steering fix cost on general capability, measured on MMBench rather than only on benign-image over-refusal?

## Method

1. Build matched triples: harmful text, the same text rendered as an image, and a benign control, so modality is the only variable. Typographic rendering is the controlled probe, not the whole visual channel.
2. Add a VLSBench leakless evaluation arm so the central claim does not rest on rendered text, which is a special and unusually easy case.
3. Measure refusal rates per modality to establish the gap exists at all.
4. Extract the refusal direction from text contrastive pairs.
5. Project image-prompt activations onto it, layer by layer.
6. Run cross-modal activation patching to localize the deficit.
7. Test whether steering along the direction closes the gap, and measure the cost on both benign-image refusal and MMBench general capability.

## Measurements

- refusal rate by modality on matched content
- refusal rate on VLSBench leakless items
- projection onto the refusal direction, per layer, per modality
- layer-wise patching effect on refusal restoration
- benign over-refusal cost and MMBench utility cost of any steering fix

## Threats to validity

- If the model simply cannot read the rendered text, a low refusal rate is an OCR failure, not a safety gap. A transcription check gates every item.
- Typographic rendering is visually encoded text, which is a narrow slice of visual harmful intent. Findings must not be generalized to the whole visual channel, which is why the VLSBench leakless arm is mandatory rather than optional.
- A modular VLM is a BASELINE for architectural contrast, never a substitute. If no unified encoder-free model can be run, the central architectural question is unanswered and the writeup must say so plainly instead of reporting modular results as if they settled it.

## Model plan

| role | choice |
|---|---|
| `unified_primary` | google/gemma-3-4b-it, which routes image patches directly into the shared decoder and is the smallest genuinely unified candidate that fits in M4 unified memory at 4-bit or bf16 with offload |
| `unified_fallback` | deepseek-ai/Janus-Pro-1B as a smaller unified alternative if Gemma exceeds memory |
| `modular_baseline` | llava-hf/llava-interleave-qwen-0.5b-hf, used strictly as the modular contrast arm |
| `escalation` | If neither unified model runs locally, the config supports a single rented-GPU run; the repo records this as a known blocker rather than quietly substituting the modular baseline. |


## Feasibility

The pilot is written for an Apple M4 with 10 cores, unified memory, the PyTorch
MPS backend, no CUDA device, and no configured API keys. Model choices are
capped accordingly (Qwen/Qwen2.5-1.5B-Instruct). The
`full` profile documents the scaled-up version of the same experiment for when a
real GPU is available, so the reduction in scale is explicit rather than hidden.

## Relationship to the posting

This proposal was checked against the mentor's verbatim posting by an
independent model before implementation began. That check, the drift it found,
and the revisions made in response are recorded in
[docs/ALIGNMENT.md](ALIGNMENT.md).
