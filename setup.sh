#!/usr/bin/env bash

gensfm_install_extension() (
    local repository=$1 revision=$2 subdirectory=${3:-.} build_dir
    build_dir=$(mktemp -d "${TMPDIR:-/tmp}/gensfm-extension.XXXXXXXX") || exit 1
    trap 'rm -rf -- "$build_dir"' EXIT
    git init -q "$build_dir/source" &&
        git -C "$build_dir/source" remote add origin "$repository" &&
        git -C "$build_dir/source" fetch --depth 1 origin "$revision" &&
        git -C "$build_dir/source" checkout -q --detach FETCH_HEAD &&
        git -C "$build_dir/source" submodule update --init --recursive &&
        python -m pip install "$build_dir/source/$subdirectory" --no-build-isolation
)

gensfm_setup_main() {
local TEMP HELP NEW_ENV BASIC TRAIN EVAL DIFFOCTREERAST VOX2SEQ
local SPCONV ERROR MIPGAUSSIAN NVDIFFRAST
local PYTORCH_VERSION PLATFORM CUDA_VERSION CUDA_MAJOR_VERSION
local CONDA_BASE CONDA_ENVIRONMENTS

# Read Arguments
TEMP=$(getopt -o h --long help,new-env,basic,train,eval,diffoctreerast,vox2seq,spconv,mipgaussian,nvdiffrast -n 'setup.sh' -- "$@") || {
    return 2
}

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
TRAIN=false
EVAL=false
DIFFOCTREERAST=false
VOX2SEQ=false
SPCONV=false
ERROR=false
MIPGAUSSIAN=false
NVDIFFRAST=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --train) TRAIN=true ; shift ;;
        --eval) EVAL=true ; shift ;;
        --diffoctreerast) DIFFOCTREERAST=true ; shift ;;
        --vox2seq) VOX2SEQ=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --mipgaussian) MIPGAUSSIAN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$#" -ne 0 ] ; then
    ERROR=true
fi

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument" >&2
    return 2
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create and activate the gensfm conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --eval                  Install evaluation dependencies"
    echo "  --diffoctreerast        Install diffoctreerast"
    echo "  --vox2seq               Install vox2seq"
    echo "  --spconv                Install spconv"
    echo "  --mipgaussian           Install mip-splatting"
    echo "  --nvdiffrast            Install nvdiffrast"
    return 0
fi

if [ "$NEW_ENV" = true ] ; then
    CONDA_BASE=$(conda info --base) || return 1
    CONDA_ENVIRONMENTS=$(conda env list --json) || return 1
    "$CONDA_BASE/bin/python" -c '
import json, pathlib, sys
environments = json.load(sys.stdin)["envs"]
if any(pathlib.Path(path).name == "gensfm" for path in environments):
    print("[ENV] The gensfm environment already exists. Activate it and omit --new-env.", file=sys.stderr)
    sys.exit(1)
' <<< "$CONDA_ENVIRONMENTS" || return 1
    conda create -n gensfm python=3.12 || return 1
    conda activate gensfm || return 1
    # Match the validated CUDA 12.8 runtime; SDPA needs no optional attention package.
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 || {
        return 1
    }
fi

# Get system information
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)") || {
    echo "[SYSTEM] Cannot import PyTorch. Activate an environment with compatible PyTorch/CUDA, or use --new-env." >&2
    return 1
}
PLATFORM=$(python -c "import torch; print('cuda' if torch.version.cuda and torch.cuda.is_available() else 'cpu')") || {
    echo "[SYSTEM] Cannot determine the PyTorch compute platform." >&2
    return 1
}
case $PLATFORM in
    cuda)
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)") || {
            echo "[SYSTEM] Cannot determine the PyTorch CUDA version." >&2
            return 1
        }
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
        ;;
    *)
        ;;
esac

if [ "$BASIC" = true ] ; then
    pip install -r requirements.txt || {
        return 1
    }
fi

if [ "$TRAIN" = true ] ; then
    pip install tensorboard==2.21.0 || return 1
fi

if [ "$EVAL" = true ] ; then
    python -c "from pytorch3d import _C; from pytorch3d.loss import chamfer_distance; from pytorch3d.ops import knn_points, corresponding_points_alignment, iterative_closest_point, sample_farthest_points, sample_points_from_meshes; from pytorch3d.ops.points_alignment import SimilarityTransform; from pytorch3d.structures import Meshes" || {
        echo "[EVAL] A compatible CUDA-enabled PyTorch3D installation is required for geometric metrics and alignment." >&2
        echo "[EVAL] Follow https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md for your PyTorch/CUDA versions, then rerun --eval." >&2
        return 1
    }
    pip install git+https://github.com/openai/CLIP.git || return 1
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        gensfm_install_extension https://github.com/NVlabs/nvdiffrast.git \
            253ac4fcea7de5f396371124af597e6cc957bfae
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM" >&2
        false
    fi || return 1
fi

if [ "$DIFFOCTREERAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        gensfm_install_extension https://github.com/hbb1/diffoctreerast.git \
            a96a4a0fc33b1d82568681e332d242110c2bfb48
    else
        echo "[DIFFOCTREERAST] Unsupported platform: $PLATFORM" >&2
        false
    fi || return 1
fi

if [ "$MIPGAUSSIAN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        gensfm_install_extension https://github.com/autonomousvision/mip-splatting.git \
            dda02ab5ecf45d6edb8c540d9bb65c7e451345a9 submodules/diff-gaussian-rasterization
    else
        echo "[MIPGAUSSIAN] Unsupported platform: $PLATFORM" >&2
        false
    fi || return 1
fi

if [ "$VOX2SEQ" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install ./extensions/vox2seq --no-build-isolation
    else
        echo "[VOX2SEQ] Unsupported platform: $PLATFORM" >&2
        false
    fi || return 1
fi

if [ "$SPCONV" = true ] ; then
    # install spconv
    if [ "$PLATFORM" = "cuda" ] ; then
        case $CUDA_MAJOR_VERSION in
            12) pip install spconv-cu126==2.3.8 ;;
            *) echo "[SPCONV] Unsupported PyTorch CUDA version: $CUDA_MAJOR_VERSION" >&2; false ;;
        esac
    else
        echo "[SPCONV] Unsupported platform: $PLATFORM" >&2
        false
    fi || return 1
fi

return 0
}

gensfm_setup_main "$@"
