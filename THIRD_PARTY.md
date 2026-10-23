# Third-party components

GenSfM code uses the [MIT License](LICENSE). Third-party code, dependencies, models, and example data retain their own licenses and notices.

| Component | Use | License / notice |
| --- | --- | --- |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Base implementation, SLat representation, and pretrained decoders | [Original MIT notice](#trellis-mit-notice) |
| [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Inria and MPII | Source retained in `trellis/renderers/gaussian_render.py`, `trellis/representations/gaussian/general_utils.py`, and `extensions/vox2seq/setup.py` | [Gaussian-Splatting license](#gaussian-splatting-license): non-commercial research and evaluation use |
| [PlenOctree](https://github.com/sxyu/plenoctree) | Spherical-harmonic utilities | [BSD 2-Clause notice retained in source](trellis/renderers/sh_utils.py) |
| Vox2Seq API by Jianfeng Xiang | Voxel serialization API declaration | MIT copyright notice retained in [`src/api.h`](extensions/vox2seq/src/api.h); [MIT terms](#trellis-mit-notice) |
| [O-CNN](https://github.com/octree-nn/ocnn-pytorch) by Peng-Shuai Wang | Z-order key utilities in `extensions/vox2seq/vox2seq/pytorch/z_order.py` | [Original MIT notice](extensions/vox2seq/vox2seq/pytorch/LICENSE) |
| [FlexiCubes fork](https://github.com/dotb1ue/FlexiCubes) | Mesh extraction; pinned Git submodule | [Apache 2.0](https://github.com/dotb1ue/FlexiCubes/blob/dcca60b3b4112d5607122f3541aa8ab4dd247e00/LICENSE.txt) |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Image encoder downloaded from upstream at runtime | [Apache 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | Differentiable rasterization | [NVIDIA license](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt) |
| [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) | Gaussian rasterizer installed by `--mipgaussian` | [Upstream license](https://github.com/autonomousvision/mip-splatting/blob/main/LICENSE.md) |
| [diffoctreerast](https://github.com/hbb1/diffoctreerast) | Optional octree rasterizer for training | [Upstream license](https://github.com/hbb1/diffoctreerast/blob/master/LICENSE) |
| [BlenderUSDZ](https://github.com/robmcrosby/BlenderUSDZ) by Robert Crosby | Bundled Blender add-on for USDZ dataset import | [GPL 3.0](dataset_toolkits/blender_script/LICENSE-USDZ); applies to `io_scene_usdz.zip` |
| Amazon Berkeley Objects | Four-view wagon example and overview illustration | [Data attribution](assets/ATTRIBUTION.md) |

The inference bundle's component notices are included with the [model repository](https://huggingface.co/dotblue/GenSfm).

## TRELLIS MIT notice

````text
    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE

````

## Gaussian Splatting license

````text
Gaussian-Splatting License
===========================

**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**.
The *Software* is in the process of being registered with the Agence pour la Protection des
Programmes (APP).

The *Software* is still being developed by the *Licensor*.

*Licensor*'s goal is to allow the research community to use, test and evaluate
the *Software*.

## 1.  Definitions

*Licensee* means any person or entity that uses the *Software* and distributes
its *Work*.

*Licensor* means the owners of the *Software*, i.e Inria and MPII

*Software* means the original work of authorship made available under this
License ie gaussian-splatting.

*Work* means the *Software* and any additions to or derivative works of the
*Software* that are made available under this License.


## 2.  Purpose
This license is intended to define the rights granted to the *Licensee* by
Licensors under the *Software*.

## 3.  Rights granted

For the above reasons Licensors have decided to distribute the *Software*.
Licensors grant non-exclusive rights to use the *Software* for research purposes
to research users (both academic and industrial), free of charge, without right
to sublicense.. The *Software* may be used "non-commercially", i.e., for research
and/or evaluation purposes only.

Subject to the terms and conditions of this License, you are granted a
non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
publicly display, publicly perform and distribute its *Work* and any resulting
derivative works in any form.

## 4.  Limitations

**4.1 Redistribution.** You may reproduce or distribute the *Work* only if (a) you do
so under this License, (b) you include a complete copy of this License with
your distribution, and (c) you retain without modification any copyright,
patent, trademark, or attribution notices that are present in the *Work*.

**4.2 Derivative Works.** You may specify that additional or different terms apply
to the use, reproduction, and distribution of your derivative works of the *Work*
("Your Terms") only if (a) Your Terms provide that the use limitation in
Section 2 applies to your derivative works, and (b) you identify the specific
derivative works that are subject to Your Terms. Notwithstanding Your Terms,
this License (including the redistribution requirements in Section 3.1) will
continue to apply to the *Work* itself.

**4.3** Any other use without of prior consent of Licensors is prohibited. Research
users explicitly acknowledge having received from Licensors all information
allowing to appreciate the adequacy between of the *Software* and their needs and
to undertake all necessary precautions for its execution and use.

**4.4** The *Software* is provided both as a compiled library file and as source
code. In case of using the *Software* for a publication or other results obtained
through the use of the *Software*, users are strongly encouraged to cite the
corresponding publications as explained in the documentation of the *Software*.

## 5.  Disclaimer

THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE *SOFTWARE* FOR COMMERCIAL PURPOSES
WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS. YOU MUST CONTACT INRIA FOR ANY
UNAUTHORIZED USE: stip-sophia.transfert@inria.fr . ANY SUCH ACTION WILL
CONSTITUTE A FORGERY. THIS *SOFTWARE* IS PROVIDED "AS IS" WITHOUT ANY WARRANTIES
OF ANY NATURE AND ANY EXPRESS OR IMPLIED WARRANTIES, WITH REGARDS TO COMMERCIAL
USE, PROFESSIONNAL USE, LEGAL OR NOT, OR OTHER, OR COMMERCIALISATION OR
ADAPTATION. UNLESS EXPLICITLY PROVIDED BY LAW, IN NO EVENT, SHALL INRIA OR THE
AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES, LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING FROM, OUT OF OR
IN CONNECTION WITH THE *SOFTWARE* OR THE USE OR OTHER DEALINGS IN THE *SOFTWARE*.

## 6.  Files subject to permissive licenses
The contents of the file ```utils/loss_utils.py``` are based on publicly available code authored by Evan Su, which falls under the permissive MIT license.

Title: pytorch-ssim\
Project code: https://github.com/Po-Hsun-Su/pytorch-ssim\
Copyright Evan Su, 2017\
License: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt (MIT)
````
