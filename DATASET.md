# Dataset preparation for GenSfM

Use this guide to prepare training and evaluation data. For reconstruction from your own photographs, follow the [inference quick start](README.md#quick-start).

GenSfM builds on the TRELLIS data pipeline below. Stage 1 learns from sparse-structure latents, per-view UV pose latents and conditioning images. Stage 2 learns from SLat targets, conditioning images and their camera transforms. The UV autoencoder must be trained or supplied before encoding the pose latents in [Step 10](#step-10-encode-uv-pose-latents-for-gensfm).

| Prepared data | Used by |
| --- | --- |
| `voxels/`, `renders/`, `features/` | Geometry and appearance preparation |
| `ss_latents/` | Stage-1 shape targets |
| `renders_cond/`, including `transforms.json` | Conditioning images and known cameras for training/evaluation |
| `suv_latents/` | Stage-1 per-view pose targets |
| `latents/` | Stage-2 SLat targets |

Training uses known rendered cameras; inference predicts them.

## Source dataset

TRELLIS-500K is a dataset of 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores.
Asset metadata is provided as CSV files; the statistics below describe the upstream TRELLIS-500K selection.

## Dataset Statistics

Filter out assets without captions if your task requires them.

| Source | Aesthetic Score Threshold | Filtered Size | With Captions |
|:-:|:-:|:-:|:-:|
| ObjaverseXL (sketchfab) | 5.5 | 168307 | 167638 |
| ObjaverseXL (github) | 5.5 | 311843 | 306790 |
| ABO | 4.5 | 4485 | 4390 |
| 3D-FUTURE | 4.5 | 9472 | 9291 |
| HSSD | 4.5 | 6670 | 6661 |
| All (training set) | - | 500777 | 494770 |
| Toys4k (evaluation set) | 4.5 | 3229 | 3180 |

## Dataset Location

Preview [TRELLIS-500K on Hugging Face](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K). The toolkits load the metadata; no manual CSV download is needed.

## Dataset Toolkits

Run the [toolkits](dataset_toolkits) from the repository root.

### Step 1: Install Dependencies

Install and activate the environment in [README.md](README.md#1-install), including training dependencies if needed. Then install the data-toolkit extras:

```bash
. ./dataset_toolkits/setup.sh
```

Rendering uses Blender 3.0.1 under `/tmp`; the scripts attempt to install it and its system libraries if absent. Install these dependencies beforehand if you lack `sudo` access. Feature extraction and latent encoding require a CUDA GPU. Follow each dataset's access requirements.

### Step 2: Load Metadata

```bash
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <OUTPUT_DIR> [--source <SOURCE>]
```

- `SUBSET`: The subset of the dataset to load. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `SOURCE`: Required if `SUBSET` is `ObjaverseXL`. Options are `sketchfab` and `github`.

For ObjaverseXL (sketchfab):

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 3: Download Data

```bash
python dataset_toolkits/download.py <SUBSET> --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to download. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.

For parallel preparation, assign each worker a unique `RANK` from `0` to `WORLD_SIZE - 1`.

This ObjaverseXL example downloads one shard out of 160000. Omit `--world_size` to download the full subset.

```bash
python dataset_toolkits/download.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --world_size 160000
```

Some datasets require Hugging Face login or manual downloads; follow the toolkit prompts.

After downloading, update the metadata file with:

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 4: Render Multiview Images

```bash
python dataset_toolkits/render.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to render. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 150.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For ObjaverseXL (sketchfab):

```bash
python dataset_toolkits/render.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

Update the metadata:

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 5: Voxelize 3D Models

```bash
python dataset_toolkits/voxelize.py <SUBSET> --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to voxelize. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For ObjaverseXL (sketchfab):

```bash
python dataset_toolkits/voxelize.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

Update the metadata:

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 6: Extract DINO Features

Extract DINO features from multiview images and aggregate them into sparse voxel grids for the SLat VAE.

```bash
python dataset_toolkits/extract_feature.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For ObjaverseXL (sketchfab):

```bash
python dataset_toolkits/extract_feature.py --output_dir datasets/ObjaverseXL_sketchfab
```

Update the metadata:

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 7: Encode Sparse Structures

Encode sparse structures into stage-1 shape targets:

```bash
python dataset_toolkits/encode_ss_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For ObjaverseXL (sketchfab):

```bash
python dataset_toolkits/encode_ss_latent.py --output_dir datasets/ObjaverseXL_sketchfab
```

Update the metadata:

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 8: Encode SLat

Encode SLat targets for stage 2:

```bash
python dataset_toolkits/encode_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For ObjaverseXL (sketchfab):

```bash
python dataset_toolkits/encode_latent.py --output_dir datasets/ObjaverseXL_sketchfab
```

Update the metadata:

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 9: Render Image Conditions

Render conditioning images with augmented views:

```bash
python dataset_toolkits/render_cond.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to render. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 24.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For ObjaverseXL (sketchfab):

```bash
python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

Update the metadata:

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 10: Encode UV pose latents for GenSfM

Complete voxelization and conditioning renders first. You need a **trained UV encoder**, its `config.json`, and its training crop/UV settings. Use the [UV autoencoder configuration](configs/vae/uv_vae_conv3d_4l8_fp16.json); this model is separate from the TRELLIS sparse-structure VAE.

The `dotblue/GenSfm` inference bundle includes a UV **decoder**, not an encoder. Supply a trained encoder or train an encoder/decoder pair with the configuration above. Use that encoder's matching decoder for generator training, evaluation, and inference; a newly trained encoder may not match the released decoder.

For a local training run, the encoder is loaded from this layout:

```text
/path/to/uv-runs/
└── uv_vae_4/
    ├── config.json
    └── ckpts/
        ├── encoder_ema0.9999_step1000000.pt
        └── decoder_ema0.9999_step1000000.pt
```

For an encoder trained with the constant UV support mask and crop ratio 1.2, run:

```bash
python dataset_toolkits/encode_suv_latent.py \
  --output_dir datasets/ObjaverseXL_sketchfab \
  --model_root /path/to/uv-runs \
  --enc_model uv_vae_4 \
  --ckpt ema0.9999_step1000000 \
  --resolution 64 \
  --const_ssuv true \
  --crop_size_ratio 1.2 \
  --batch_size 8 \
  --num_workers 4

python dataset_toolkits/build_metadata.py ObjaverseXL \
  --output_dir datasets/ObjaverseXL_sketchfab
```

Match these settings to the encoder's training configuration. The default encoder input combines one support-mask channel and two UV channels; do not enable a depth channel for a three-channel checkpoint. `--enc_pretrained` is an alternative for an encoder exported in the format accepted by `trellis.models.from_pretrained`; it is not a path to an arbitrary training `.pt` file.

The example writes per-object archives under `suv_latents/uv_vae_4_ema0.9999_step1000000/`. Set `dataset.args.suv_latent_model` and the corresponding evaluation field in the stage-1 configuration to that directory name. The multiview generator configurations default to the released compact decoder through `pretrained_suv_dec`, with `suv_latent_channels=8`. When using a different trained encoder, replace `pretrained_suv_dec` and `eval_models.suv_decoder.path` with the matching decoder prefix in the format accepted by `trellis.models.from_pretrained`. Alternatively, set dataset `suv_dec_path`/`suv_dec_ckpt` to its local training run, and configure the evaluation decoder with that run's decoder `name`, `args`, and `.pt` checkpoint `path`. Crop-dependent keys inside each archive must match the crop settings used for training.

Stage 1 needs `ss_latents/` from Step 7 and these UV pose latents. Stage 2 uses `latents/` from Step 8 and conditioning images/cameras from Step 9. See [training and evaluation](README.md#training-and-evaluation) for the generator configurations.

## Training checkpoint inputs

Model-prefix fields such as `pretrained_ss_dec`, `pretrained_suv_dec`, and `eval_models.*.path` without a model `name` accept Hugging Face prefixes or local `.json`/`.safetensors` prefixes. The multiview stage-1 evaluation configurations use the released ConvNeXt stage-2 model at `dotblue/GenSfm/ckpts/stage2_slat`, including its matching SLat encoder dependency and latent normalization.

Raw weight fields, including `trainer.args.finetune_ckpt`, `evaluator.args.pretrained_ckpt`, and a named evaluation model's `path`, read local checkpoint files. Configurations that initialize from upstream TRELLIS use `checkpoint/TRELLIS-image-large/ckpts/`; download those public files before training or evaluation:

```bash
hf download microsoft/TRELLIS-image-large \
  --include 'ckpts/*' \
  --local-dir checkpoint/TRELLIS-image-large
```

Match the [training configurations](README.md#training-and-evaluation) to your latent directory names and UV encoder/decoder pair. Use `--override` in `train.py` and `eval.py` to set dataset paths, checkpoint inputs, and batch sizes.
