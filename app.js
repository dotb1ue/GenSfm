/* A buildless, data-driven research page. Scientific copy lives in data/. */
import { createViewer } from './viewer.mjs?v=20261023-flight1';
import { renderResults } from './results.mjs';

const $ = (id) => document.getElementById(id);
const state = { items: [], selectedId: null, dataset: 'All', viewer: null, viewerPromise: null, ready: false, selection: 0, framing: 'context', sidebarReady: false };
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function url(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const resolved = new URL(value, document.baseURI);
    return ["http:", "https:"].includes(resolved.protocol) ? resolved.href : null;
  } catch { return null; }
}

function setLink(element, value) {
  const target = url(value);
  if (!target) return false;
  element.href = target;
  if (new URL(target).origin !== location.origin) {
    element.target = "_blank";
    element.rel = "noopener noreferrer";
  }
  return true;
}

function image(source, alt, options = {}) {
  const img = node("img");
  const resolved = url(source);
  if (resolved) img.src = resolved;
  img.alt = alt;
  img.width = options.width || 500;
  img.height = options.height || 500;
  if (options.lazy !== false) img.loading = "lazy";
  img.decoding = "async";
  return img;
}

async function loadJSON(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function renderContent(content) {
  if (!content || !content.title || !content.projectName) throw new Error("Paper title and project name are missing.");
  document.title = `${content.projectName} — ${content.title}`;
  $("project-name").textContent = content.projectName;
  $("footer-brand").textContent = content.projectName;
  $("project-kicker").textContent = content.projectName;
  $("paper-title").textContent = content.title;
  if (content.subtitle) {
    $("paper-subtitle").textContent = content.subtitle;
    $("paper-subtitle").hidden = false;
    document.querySelector('meta[name="description"]').content = content.subtitle;
  }
  if (content.venue) {
    $("paper-status").textContent = content.venue;
    $("paper-status").hidden = false;
  }
  const notes = new Set();
  (content.authors || []).forEach((author) => {
    const wrapper = node("span", "author");
    const label = author.url ? node("a", null, author.name) : node("span", null, author.name);
    if (author.url && !setLink(label, author.url)) label.removeAttribute("href");
    wrapper.append(label);
    const superscript = (author.affiliations || []).join(",") + (author.noteMarker || "");
    if (superscript) wrapper.append(node("sup", null, superscript));
    if (author.note) notes.add(author.note);
    $("authors").append(wrapper);
  });
  (content.affiliations || []).forEach((affiliation) => {
    const line = node("span");
    if (affiliation.id) line.append(node("sup", null, affiliation.id));
    line.append(document.createTextNode(affiliation.name));
    $("affiliations").append(line);
  });
  if (notes.size) {
    $("author-notes").textContent = Array.from(notes).join(" · ");
    $("author-notes").hidden = false;
  }
  (content.resources || []).forEach((resource) => {
    if (resource.pending) {
      const placeholder = node("span", "resource-link is-pending", resource.label);
      placeholder.setAttribute("aria-disabled", "true");
      placeholder.title = "The arXiv link will be added when available.";
      $("resources").append(placeholder);
      return;
    }
    const link = node("a", "resource-link");
    if (!setLink(link, resource.url)) return;
    link.append(node("span", null, resource.label), node("span", null, "↗"));
    link.lastChild.setAttribute("aria-hidden", "true");
    $("resources").append(link);
  });
  $("abstract").textContent = content.abstract || "";
  (content.contributions || []).forEach((contribution) => {
    const row = node("div", "contribution");
    const copy = node("div");
    copy.append(node("h3", null, contribution.title), node("p", null, contribution.description));
    row.append(copy);
    $("contributions").append(row);
  });
  if (content.method && url(content.method.image)) {
    $("method").hidden = false;
    $("method-image").src = url(content.method.image);
    $("method-image").alt = content.method.alt || "Overview of the reconstruction method";
    setLink($("method-image-link"), content.method.image);
    $("method-caption").textContent = content.method.caption || "";
  } else $("method-nav").hidden = true;
  renderFigures(content.figures || []);
  renderResults(content.results);
  if (content.citation) {
    $("citation").hidden = false;
    $("bibtex").textContent = content.citation;
  } else $("citation-nav").hidden = true;
  if (content.footerNote) $("footer-note").textContent = content.footerNote;
}

function renderFigures(figures) {
  const container = $("paper-figures");
  figures.forEach((figure) => {
    if (!url(figure.image)) return;
    const section = node("section", "paper-figure-section section-shell");
    if (figure.id && /^[a-zA-Z0-9_-]+$/.test(figure.id)) section.id = figure.id;
    if (figure.title) {
      const heading = node("div", "section-heading");
      heading.append(node("h2", null, figure.title));
      section.append(heading);
    }
    const frame = node("figure", "method-figure");
    const link = node("a");
    link.title = "Open full-size figure";
    setLink(link, figure.image);
    link.append(image(figure.image, figure.alt || figure.title || "Paper result figure", { width: 1400, height: 700 }));
    frame.append(link);
    if (figure.caption) frame.append(node("figcaption", null, figure.caption));
    section.append(frame);
    container.append(section);
  });
}

function visibleItems() {
  return state.items.filter(item => state.dataset === 'All' || item.domain === state.dataset);
}

function renderFilters() {
  $('dataset-filters').replaceChildren();
  for (const [key, label] of [['All', 'All objects'], ['synthetic', 'Synthetic'], ['real-world', 'Real-world']]) {
    const button = node('button', null, label);
    button.type = 'button'; button.dataset.filter = key;
    button.setAttribute('aria-pressed', String(state.dataset === key));
    button.addEventListener('click', () => {
      state.dataset = key; renderFilters(); renderObjectList();
      const visible = visibleItems();
      if (!visible.some(item => item.id === state.selectedId)) selectObject(visible[0]?.id);
      $('dataset-filters').querySelector(`[data-filter="${key}"]`)?.focus({ preventScroll: true });
    });
    $('dataset-filters').append(button);
  }
}

function renderObjectList() {
  $('object-list').replaceChildren();
  const items = visibleItems();
  $('object-count').textContent = `${items.length} objects`;
  for (const item of items) {
    const button = node('button', 'object-option'); button.type = 'button'; button.dataset.objectId = item.id;
    button.setAttribute('aria-label', `Explore ${item.label}`); button.setAttribute('aria-pressed', String(item.id === state.selectedId));
    button.append(image(item.thumbnail, '', { width: 180, height: 180 }), node('span', null, item.label));
    button.addEventListener('click', () => selectObject(item.id));
    $('object-list').append(button);
  }
}

function updateViewerUI(snapshot = {}) {
  if (snapshot.ready && (snapshot.caseId !== state.selectedId || !state.sidebarReady)) return;
  if (snapshot.ready !== undefined) state.ready = snapshot.ready;
  const native = snapshot.mode === 'view', view = snapshot.viewIndex;
  const moving = Boolean(snapshot.transitioning);
  $('viewer-canvas').setAttribute('aria-busy', String(moving));
  if (snapshot.error) {
    $('viewer-status').hidden = false; $('viewer-status').classList.add('is-error');
    $('viewer-status-copy').textContent = 'The 3D scene could not load. Select another object or refresh to retry.';
  } else {
    $('viewer-status').hidden = state.ready;
    $('viewer-status').classList.remove('is-error');
  }
  $('viewer-mode').textContent = native ? `Reprojection · View ${view + 1}` : '3D Context';
  $('viewer-hint').textContent = native ? 'Original predicted camera' : 'Drag to rotate · Scroll to zoom · Double-click a camera';
  $('input-opacity').disabled = !native || !state.ready || moving;
  $('overlay-control').classList.toggle('is-disabled', !native || moving);
  $('return-context').disabled = !native || !state.ready;
  $('gt-camera-control').hidden = !snapshot.hasReferenceCameras;
  $('reference-note').hidden = !snapshot.hasReferenceCameras;
  $('show-cameras').disabled = native || !state.ready || moving;
  $('show-gt-cameras').disabled = native || !state.ready || moving;
  for (const button of $('input-thumbnails').querySelectorAll('button')) {
    button.disabled = !state.ready;
    button.setAttribute('aria-pressed', String(native && Number(button.dataset.view) === view));
  }
  for (const [id, framing] of [['fit-context', 'context'], ['fit-object', 'object']]) {
    $(id).disabled = !state.ready;
    $(id).setAttribute('aria-pressed', String(!native && state.framing === framing));
  }
}

async function selectObject(id) {
  const item = state.items.find(candidate => candidate.id === id);
  if (!item) return;
  const selection = ++state.selection;
  state.selectedId = id; state.ready = false; state.sidebarReady = false; state.framing = 'context';
  $('object-detail').hidden = false;
  $('object-title').textContent = item.label;
  $('object-source').textContent = item.domain === 'real-world' ? 'REAL-WORLD CAPTURE' : `${item.dataset === 'Toys4k' ? 'TOYS4K' : item.dataset} / SYNTHETIC`;
  $('object-protocol').textContent = '4 input views · Textured reconstruction';
  $('viewer-status-copy').textContent = `Loading ${item.label.toLowerCase()}…`;
  $('viewer-status').hidden = false;
  $('input-thumbnails').replaceChildren(); updateViewerUI({ready:false});
  for (const entry of $('object-list').querySelectorAll('button')) entry.setAttribute('aria-pressed', String(entry.dataset.objectId === id));
  try {
    const manifestURL = url(item.manifest);
    if (!state.viewerPromise) state.viewerPromise = createViewer({container:$('viewer-canvas'),onState:updateViewerUI}).catch(error=>{state.viewerPromise=null;throw error;});
    state.viewer = await state.viewerPromise;
    window.gensfmViewer = state.viewer;
    if (selection !== state.selection) return;
    // Invalidate a previous case immediately, before fetching sidebar metadata.
    const loading = state.viewer.loadCase(manifestURL);
    const manifest = await loadJSON(manifestURL);
    if (selection !== state.selection) return;
    manifest.images.forEach((input, view) => {
      const button = node('button', 'input-thumb'); button.type = 'button'; button.dataset.view = String(view);
      button.disabled = true; button.setAttribute('aria-pressed', 'false');
      button.setAttribute('aria-label', `Look through recovered camera ${view + 1}`);
      button.style.setProperty('--view-color', ['#438c96','#bd913f','#8c78ad','#b66e89'][view]);
      button.append(image(new URL(input.file, manifestURL).href, `Observed input view ${view + 1}`, { width: input.width, height: input.height, lazy:false }), node('span', null, `View ${view + 1}`));
      button.addEventListener('click', () => { state.viewer?.setView(view); state.viewer?.setOverlay(Number($('input-opacity').value)/100); });
      $('input-thumbnails').append(button);
    });
    if (selection !== state.selection) return;
    state.sidebarReady = true;
    await loading;
    if (selection !== state.selection) return;
    state.viewer.setOverlay(Number($('input-opacity').value)/100);
    state.viewer.setCameras($('show-cameras').checked);
    state.viewer.setReferenceCameras($('show-gt-cameras').checked);
    state.viewer.setCube(true);
    $('gallery-status').textContent = '';
  } catch (error) {
    if (selection !== state.selection) return;
    console.error('Reconstruction could not load:',error);updateViewerUI({ready:false,error:String(error)});
  }
}

function frameScene(fitCameras) {
  state.framing = fitCameras ? 'context' : 'object';
  state.viewer?.orbit({fitCameras});
}

function renderGallery(gallery) {
  if (!gallery.items?.length) throw new Error('The interactive gallery is empty.');
  state.items = gallery.items;
  const identities = new Set(state.items.map(item=>item.id));
  if (identities.size !== state.items.length) throw new Error('Duplicate object identity.');
  const video = $('paper-video'); video.src = url(gallery.video.src); video.poster = url(gallery.video.poster);
  setLink($('video-download'), gallery.video.src);
  renderFilters();renderObjectList();selectObject(gallery.defaultCaseId || state.items[0].id);
}

$('fit-context').addEventListener('click',()=>frameScene(true));
$('fit-object').addEventListener('click',()=>frameScene(false));
$('return-context').addEventListener('click',()=>state.viewer?.orbit({restore:true}));
$('show-cameras').addEventListener('change',event=>state.viewer?.setCameras(event.target.checked));
$('show-gt-cameras').addEventListener('change',event=>state.viewer?.setReferenceCameras(event.target.checked));
$('input-opacity').addEventListener('input',event=>{
  const value=Number(event.target.value);$('opacity-value').value=`${value}%`;
  event.target.setAttribute('aria-valuetext',`${value} percent input image`);state.viewer?.setOverlay(value/100);
});
for (const [id,direction] of [['case-prev',-1],['case-next',1]]) $(id).addEventListener('click',()=>$('object-list').scrollBy({left:direction*$('object-list').clientWidth*.8,behavior:reducedMotion?'instant':'smooth'}));

$("copy-citation").addEventListener("click", async () => {
  const text = $("bibtex").textContent;
  try {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
    else {
      const field = node("textarea");
      field.value = text;
      field.style.position = "fixed";
      field.style.left = "-9999px";
      document.body.append(field);
      field.select();
      const copied = document.execCommand("copy");
      field.remove();
      if (!copied) throw new Error("Clipboard access is unavailable.");
    }
    $("copy-citation").textContent = "Copied ✓";
    $("copy-status").textContent = "BibTeX copied to the clipboard.";
    window.setTimeout(() => { $("copy-citation").textContent = "Copy BibTeX ↗"; }, 2200);
  } catch {
    const range = document.createRange();
    range.selectNodeContents($("bibtex"));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    $("copy-status").textContent = "Clipboard access is unavailable. The citation is selected; use your browser's Copy command.";
    $("copy-citation").textContent = "Citation selected";
  }
});



Promise.all([loadJSON('data/content.json'),loadJSON('data/interactive-gallery.json')])
  .then(([content,gallery])=>{renderContent(content);renderGallery(gallery);})
  .catch(error=>{
    console.error('Project page could not load:',error);
    $('gallery-status').textContent='The examples could not load. Refresh the page to try again.';
    const notice=node('p','load-error','The project data is temporarily unavailable. Please refresh the page.');notice.setAttribute('role','alert');$('main').prepend(notice);
  });
