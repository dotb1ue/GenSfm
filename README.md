# GenSfM project website

[Project page](https://dotb1ue.github.io/GenSfm/) · [Code](https://github.com/dotb1ue/GenSfm) · [Models](https://huggingface.co/dotblue/GenSfm)

A static research page with the paper video, 36 interactive reconstructions, method figures, benchmark results, and a copyable citation. No build step or backend is required. JavaScript dependencies, fonts, and gallery assets are served locally.

## Preview

From this directory, start an HTTP server:

```bash
python -m http.server 5180
```

Open `http://localhost:5180/`. JSON and ES modules require HTTP; opening `index.html` directly from disk will not work. All runtime paths are relative, including when the page is hosted under a project subdirectory.

## Content

- `data/content.json`: title, authors, affiliations, resource links, method figures, results, and BibTeX.
- `data/interactive-gallery.json`: video, object order, filters, and per-case manifests.
- `assets/interactive/`: textured meshes, four input images per object, thumbnails, and camera data.
- `assets/video/`, `assets/derived/`: video, poster, and manuscript figures.
- `app.js`, `styles.css`: page content, gallery controls, and responsive presentation.
- `viewer.mjs`, `projection.mjs`: interactive camera views and reprojection.
- `results.mjs`, `results.css`: benchmark charts and full tables.

The Code and Models buttons use the project repositories. The arXiv entry is a non-clickable placeholder until the paper identifier is available. To activate it, replace its `pending` field with the confirmed `url` in `data/content.json`. Add the same confirmed paper URL to the BibTeX entry when appropriate.

## Interactive gallery

Drag to rotate and scroll to zoom. Click an input image or double-click a recovered camera to move smoothly into that view. Adjust Input overlay to compare the observed image with its reconstruction; Back to 3D context restores the previous free view. Context and Object control framing, and the camera toggles show recovered and available reference cameras.

Meshes and inputs are unchanged model outputs. Reference cameras are aligned for display; reprojection uses the predicted cameras. Asset credits and coordinate-display notes are in [ASSETS.md](ASSETS.md).

Local export tools and historical source records live in ignored `notes/`; local checks and screenshots remain in ignored `tests/` and `.qa/`. They are not runtime dependencies.

## Hosting

The site is published from the root of the `gh-pages` branch using GitHub Pages. `.nojekyll` keeps it a plain static site. Code can be maintained independently on the repository's default branch. Push website updates to `gh-pages` after previewing them locally.
