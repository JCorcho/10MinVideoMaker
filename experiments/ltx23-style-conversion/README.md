# Preserved LTX 2.3 Style-Conversion Leads

These frozen API workflows preserve two unexpected results from continuation
acceptance run `continuation-acceptance-20260731-065935`, source job
`20260730-0217`, scene 1. They have no production routing effect. Both methods
were rejected for ordinary same-style continuation and automatic continuation
remains locked.

The observations come from one test scene. They are reproducible research leads,
not a claim that either method generalizes or is ready for production.

## Shared settings

- Checkpoint: `10Eros_v1.4_fp8mixed_learned.safetensors`
- Text encoder: `gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors`
- Mandatory LoRAs: `LTX2.3_DMD_reshaped_r256.safetensors` at `1.0` and
  `JoyAI-Echo-content_r256.safetensors` at `0.5`
- Sampler: LCM, CFG `1.0`, seed `14005304055458111461`
- Stage-one sigmas:
  `1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0`
- Stage-two sigmas: `0.909375, 0.725, 0.421875, 0`
- Spatial upscaler:
  `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`, tiled at size 11,
  overlap 6
- Output: 768×1344, 24 fps, lossless FFV1 source window

## Decoded 17-frame guide

Stage one makes a 121-frame 384×672 continuation from the source final image,
then saves its last 16 latent tokens. Stage two decodes frames 96–112 from the
base window, inserts those 17 RGB frames as a strength-1.0 guide at frame 0,
crops guide conditioning, spatially upscales, and performs the second LCM pass.

Human review found that this retained the scene geometry, wardrobe design,
motion direction, and camera movement while converting the cel-shaded anime/game
render to a photoreal live-action look. It also corrected a visible hand defect
in this sample. That is unacceptable for same-style continuation but is the
stronger lead for a future opt-in anime-to-live-action converter.

For a production-faithful seam preview, retain base frames 0–112 and append
continuation frame 17 onward. Frames 0–16 are conditioning overlap.

## Latent 25-frame overlap

Stage one loads the preceding stage-one latent and uses `LTXVExtendSampler` with
24-frame overlap, 97 new frames, and strength `0.5`, then saves the last 17
latent tokens. Stage two decodes base frames 96–120, inserts the 25 RGB guide
frames at latent frame index 8 with strength `1.0`, crops guide conditioning,
spatially upscales, and performs the second LCM pass.

Human review found a semi-realistic 3D conversion rather than style continuity.
The hand defect improved, but forward walking stopped, remaining movement became
robotic, and the hair became hijab-like. This is a weaker future
anime-to-semi-realistic-3D lead and is not a production continuation method.

For a production-faithful seam preview, retain base frames 0–120 and append
continuation frame 25 onward. Frames 0–24 are conditioning overlap.

## Audit and immutability

`manifest.json` binds every frozen workflow to its SHA-256 and records the
D-drive paths and hashes of the large raw videos. Raw media is deliberately not
duplicated into Git or onto C:. Any intentional experiment derived from these
files should copy them to a new folder and leave this evidence unchanged.
