# Assets and credits

The gallery contains 36 textured reconstructions with their four observed inputs and recovered cameras. Meshes, textures, inputs, and predicted camera matrices are unchanged inference outputs. Display thumbnails are rendered from those meshes.

The page includes synthetic examples from Toys4K and ABO, alongside real-world captures. Synthetic reference cameras use one similarity alignment for display in the reconstruction frame; they do not modify the reconstruction or its predicted cameras. Real-world examples without ground-truth cameras omit that overlay. Reprojection always uses the recovered camera matrices and complete crop intrinsics.

The manuscript figures and video are GenSfM research materials. The publication includes browser-ready manuscript figures, the paper video, and the interactive examples. Observed inputs and reconstructed meshes retain their exported content.

Third-party runtime dependencies and their license notices are included locally:

- [Three.js](https://github.com/mrdoob/three): MIT; see `assets/vendor/three/LICENSE`.
- [Inter](https://github.com/rsms/inter): SIL Open Font License; see `assets/fonts/Inter-LICENSE.txt`.

The presentation was informed by [CUPID](https://cupid3d.github.io/), [StructParts](https://zhao-yiqun.github.io/structparts/), [Nerfies](https://nerfies.github.io/), and [Mip-NeRF 360](https://jonbarron.info/mipnerf360/). Their visual assets and source code are not included.
