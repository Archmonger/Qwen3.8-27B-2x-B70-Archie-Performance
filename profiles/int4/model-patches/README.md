Placeholder.

Will hold the corrected `drafter-config.json` for the INT4 profile. It is expected
to be identical to `../../fp8/model-patches/drafter-config.json`, because the
drafter's contract with the target is hidden-state taps (`target_layer_ids`,
`num_target_layers`, `hidden_size`, `vocab_size`) that the INT4 text config matches
exactly — weight quantization does not change the activations the projector
consumes.

Until the INT4 profile is implemented this directory holds no config file, only
this note. See `../README.md`.
