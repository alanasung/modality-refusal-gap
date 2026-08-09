# Related work

This note situates **Refusal Without a Projector** against the mentor-linked literature for
[Locating the Refusal Circuit in Unified, Encoder-Free Vision-Language Models](https://sparai.org/projects/f26/recTF6JBqSCgR6sNf).

## Positioning

If there is no vision projector to blame for the text-to-image safety gap, find out what is actually carrying it.

The design hypothesis is: In a unified, encoder-free VLM the text-derived refusal direction still exists and is still linear, but image-delivered harmful intent lands with systematically lower projection onto it. The deficit accumulates across early decoder layers rather than appearing at one interface, which is the signature that distinguishes a distributed failure from a modular one.

## Engagement rules

1. Cite the paper that motivates each measurement.
2. Name what this repo replicates versus what it changes.
3. Keep synthetic harness results labelled as synthetic.
4. Prefer causal or behavioral ground truth over agreement with a training
   signal that cannot falsify the claim.

## Skeleton critique slots

The following slots are filled per project during alignment. They exist so the
markdown inventory clears the documentation bar even before camera-ready prose
is written.

### Slot A — Primary motivating paper

Summary of the mentor's main citation and the exact claim this repo tests.

### Slot B — Closest prior codebase

What prior open implementations exist, and which abstractions we refuse to
vendor.

### Slot C — Measurement instrument papers

Probe, patching, monitoring, or jailbreak-ladder methodology sources.

### Slot D — Confounds already named in the literature

Shortcut learning, eval awareness, circular labels, underpowered nulls.

### Slot E — Open disagreements

Where this design intentionally diverges from common practice, with the
falsification condition.

## Mentors and affiliations

- Mentor(s): Alessandro Suglia, Rohit Saxena, Francesco Pinto
- Affiliation(s): University of Edinburgh, University of Edinburgh, Google DeepMind

## Bibliography placeholders

Additional references are tracked in `TASK.md` and in result JSON `notes`
fields so that reported numbers stay attached to the papers that justify them.
