# All-Speaker Training

The original SemTalk release and its default configs remain the Speaker 2
protocol. This directory adds a separate 25-speaker workflow without changing
those defaults.

The all-speaker protocol uses the available English BEAT2 speakers:

```text
1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18,
20, 21, 22, 23, 24, 25, 27, 28, 30
```

## Released layout

From the SemTalk repository root, install the all-speaker archive with:

```shell
unzip SemTalk_all_speakers_weights_25spk.zip
cp -R SemTalk_all_speakers_weights/weights/all_speakers ./weights/
```

The resulting files are organized as:

```text
weights/all_speakers/
├── pretrained_vq/
│   ├── rvq_face_600.bin
│   ├── rvq_hands_500.bin
│   ├── rvq_upper_500.bin
│   ├── rvq_lower_600.bin
│   └── last_1700_foot.bin
├── best_semtalk_base.bin
└── best_semtalk_sparse.bin
```

`SEMTALK_PRETRAINED_VQ_DIR` selects this all-speaker RVQ/VAE suite. If the
variable is not set, SemTalk continues to use the original Speaker 2 directory
at `weights/pretrained_vq`.

## Dataset

```shell
scripts/all_speakers/run.sh prepare-train
scripts/all_speakers/run.sh prepare-test
```

By default this writes the all-speaker caches to `datasets/all_speakers`.
Set `SEMTALK_DATA_ROOT` or `SEMTALK_ALL_SPEAKER_DATASET_ROOT` to use other
locations.

## Training

Train the five motion representation models:

```shell
scripts/all_speakers/run.sh vq-face
scripts/all_speakers/run.sh vq-hands
scripts/all_speakers/run.sh vq-upper
scripts/all_speakers/run.sh vq-lower
scripts/all_speakers/run.sh vae-global
```

These commands use the released training lengths: 600 epochs for face, 500
for hands, 500 for upper body, 600 for lower body, and 1700 for global/root
motion. `SEMTALK_VQ_SEED` defaults to `2021`.

Select the desired checkpoints and place them in
`weights/all_speakers/pretrained_vq` using the released filenames. Then train
the two SemTalk stages:

```shell
scripts/all_speakers/run.sh base
scripts/all_speakers/run.sh sparse
```

Base and Sparse each run for 400 epochs. `SEMTALK_SEED` selects the random seed
and defaults to `43`, the seed of both released checkpoints. The model search
also evaluated seeds `42` and `44`; set a different output root or retain each
seed directory when reproducing all three runs.

Sparse Motion Generation must be initialized with the selected all-speaker
Base checkpoint at `weights/all_speakers/best_semtalk_base.bin`.

The released run trained three random seeds, evaluated checkpoints every 10
epochs, and then resumed around each coarse minimum to evaluate every local
epoch. The released Base and Sparse files are the lowest-FGD checkpoints after
that two-stage search. Details and checksums are in
`results/all_speakers/best_run.json` and
`results/all_speakers/SHA256SUMS`.

## Evaluation

```shell
scripts/all_speakers/run.sh test
```

The default test implementation writes paired `gt_*.npz` and `res_*.npz`
files. The complete released NPZ set is also available from the link in the
main README.
