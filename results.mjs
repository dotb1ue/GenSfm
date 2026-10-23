const byId = id => document.getElementById(id);
const node = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};
const metrics = {
  generation: ['psnr', 'f01', 'lpips'],
  reconstruction: ['psnr', 'f01', 'lpips'],
  poses: ['rre', 'acc5', 'rte'],
};
const metricNames = {
  f01: 'F-score @ 0.01',
  acc5: 'Acc @ 5° (%)',
};

function numericValue(raw) {
  if (typeof raw === 'number') return Number.isFinite(raw) && raw >= 0 ? raw : null;
  if (typeof raw !== 'string' || !/^\d+(?:\.\d+)?%?$/.test(raw.trim())) return null;
  return Number(raw.trim().replace('%', ''));
}

function axisMaximum(key, values) {
  if (key === 'f01') return 1;
  if (key === 'acc5') return 100;
  const largest = Math.max(0, ...values.filter(value => value !== null));
  if (!largest) return 1;
  const step = 10 ** Math.floor(Math.log10(largest));
  return Number((Math.ceil(largest / step) * step).toPrecision(10));
}

function chart(table, dataset, rows, column) {
  const name = metricNames[column.key] || column.label;
  const direction = column.direction === 'higher' ? 'Higher is better' : 'Lower is better';
  const arrow = column.direction === 'higher' ? '↑' : '↓';
  const values = rows.map(row => numericValue(row.values?.[column.key]));
  const maximum = axisMaximum(column.key, values);
  const card = node('section', 'metric-card');
  card.dataset.metric = column.key;
  card.setAttribute('role', 'img');
  card.setAttribute('aria-label', `${table.label}, ${dataset}. ${name}. ${direction}. ` +
    rows.map((row, index) => `${row.label}: ${values[index] === null ? 'not reported' : row.values[column.key]}`).join('; ') +
    `. Axis from 0 to ${maximum}${column.key === 'acc5' ? ' percent' : ''}.`);
  const heading = node('div', 'metric-card-heading');
  heading.append(node('h3', null, `${name} ${arrow}`), node('p', 'metric-preference', direction));
  card.append(heading);
  const bars = node('div', 'metric-bars');
  rows.forEach((row, index) => {
    const value = values[index];
    const item = node('div', `metric-row${row.highlight ? ' metric-row-highlight' : ''}`);
    item.dataset.method = row.label;
    item.dataset.metric = column.key;
    item.dataset.value = value === null ? 'missing' : String(value);
    const labels = node('div', 'metric-row-labels');
    labels.append(node('span', 'metric-method', row.label), node('span', 'metric-value', row.values?.[column.key] ?? '—'));
    item.append(labels);
    if (value !== null) {
      const track = node('div', 'metric-track');
      const fill = node('div', 'metric-fill');
      fill.style.width = `${Math.min(100, value / maximum * 100)}%`;
      track.append(fill);
      item.append(track);
    } else item.append(node('p', 'metric-missing', 'Not reported'));
    bars.append(item);
  });
  const axis = node('div', 'metric-axis');
  axis.append(node('span', null, '0'), node('span', null, `${maximum}${column.key === 'acc5' ? '%' : ''}`));
  card.append(bars, axis);
  return card;
}

function renderFullTable(table, dataset, rows) {
  const target = byId('results-table');
  target.querySelector('caption').textContent = `${table.label} · ${dataset}`;
  const headings = node('tr');
  const method = node('th', null, table.rowLabel || 'Method');
  method.scope = 'col';
  headings.append(method);
  table.columns.forEach(column => {
    const heading = node('th', null, column.label);
    heading.scope = 'col';
    if (column.direction) {
      const direction = node('span', 'metric-direction', column.direction === 'higher' ? '↑' : '↓');
      direction.setAttribute('aria-label', column.direction === 'higher' ? 'higher is better' : 'lower is better');
      heading.append(direction);
    }
    headings.append(heading);
  });
  target.querySelector('thead').replaceChildren(headings);
  const body = rows.map(row => {
    const tr = node('tr', row.highlight ? 'highlight' : '');
    const label = node('th', null, row.label);
    label.scope = 'row';
    tr.append(label);
    table.columns.forEach(column => tr.append(node('td', null, row.values?.[column.key] ?? '—')));
    return tr;
  });
  target.querySelector('tbody').replaceChildren(...body);
}

export function renderResults(results) {
  const tables = (results?.tables || []).filter(table => metrics[table.id] && table.columns?.length && table.rows?.length);
  if (!tables.length) return;
  let selected = tables.find(table => table.id === 'generation') || tables[0];
  let dataset = 'Toys4K';
  const protocols = byId('result-tabs');
  const datasets = byId('result-datasets');
  protocols.replaceChildren();
  datasets.replaceChildren();
  protocols.setAttribute('role', 'group');
  protocols.setAttribute('aria-label', 'Evaluation protocol');
  datasets.setAttribute('role', 'group');
  datasets.setAttribute('aria-label', 'Evaluation dataset');
  byId('full-results').open = false;
  byId('quantitative').hidden = false;

  function update() {
    const rows = selected.rows.filter(row => row.values?.dataset === dataset);
    const chartRows = selected.id === 'poses'
      ? rows.filter(row => row.label !== 'GenSfM (known intrinsics)') : rows;
    byId('results-intro').textContent = [results.intro, selected.intro].filter(Boolean).join(' ');
    byId('results-note').textContent = [selected.note, results.note].filter(Boolean).join(' ');
    const note = byId('results-chart-note');
    note.textContent = selected.id === 'poses'
      ? 'Charts use predicted intrinsics for GenSfM. The full table also includes the auxiliary known-intrinsics condition.' : '';
    note.hidden = selected.id !== 'poses';
    protocols.querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.protocol === selected.id)));
    datasets.querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.dataset === dataset)));
    byId('result-charts').replaceChildren(...metrics[selected.id].map(key =>
      selected.columns.find(column => column.key === key)).filter(Boolean).map(column =>
      chart(selected, dataset, chartRows, column)));
    renderFullTable(selected, dataset, rows);
  }

  tables.forEach(table => {
    const button = node('button', null, table.label);
    button.type = 'button';
    button.dataset.protocol = table.id;
    button.addEventListener('click', () => { selected = table; update(); });
    protocols.append(button);
  });
  ['Toys4K', 'GSO'].forEach(name => {
    const button = node('button', null, name);
    button.type = 'button';
    button.dataset.dataset = name;
    button.addEventListener('click', () => { dataset = name; update(); });
    datasets.append(button);
  });
  update();
}
