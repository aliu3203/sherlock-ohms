import {SEED_REFERENCE, createLUT, normalizeLUT, makeDemoResult, validateReference, partStatus} from './demo.mjs';
import {normalizeReference, normalizeHistory, normalizeCalibration, normalizeInspection, eventProgress} from './api.mjs';

// Override before this module loads via window.SHERLOCK_CONFIG; see API_CONTRACT.md.
const config = window.SHERLOCK_CONFIG || {};
const API = {
  references: '/api/references', history: '/api/inspections', inspect: '/api/inspect',
  calibration: '/api/calibration', calibrate: '/api/calibrate',
  events: id => `/api/events/${encodeURIComponent(id)}`,
  inspection: id => `/api/inspections/${encodeURIComponent(id)}`,
  ...config.routes,
};
const $ = id => document.getElementById(id);
const COLORS = {valid: '#237e45', critical: '#db3838', warning: '#bb7309', uncertain: '#6b7280'};
const STORE = 'sherlock-ohms-demo-v1';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(STORE) || '{}'); } catch { /* Storage may be unavailable. */ }
const state = {
  demo: true, scenario: 'exact', busy: false, generation: 0, view: 'inspect', result: null,
  references: [structuredClone(SEED_REFERENCE)], history: [], calibration: null,
  demoReferences: Array.isArray(saved.references) ? saved.references.filter(r => { try { validateReference(r); return typeof r.id === 'string' && r.id !== SEED_REFERENCE.id; } catch { return false; } }) : [],
  demoHistory: Array.isArray(saved.history) ? saved.history.filter(h => h?.simulation === true && h?.verdict && typeof h.id === 'string').slice(0, 50) : [],
  selectedFinding: null, source: null, controller: null, timers: new Set(), requestId: null,
  lut: createLUT(), start: 0, calibrationBusy: false,
};
function element(tag, text, className) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; }
function svg(tag, attrs = {}) { const el = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v)); return el; }
function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function setConnection(text, offline = false) { $('connection-status').textContent = text; $('connection-dot').classList.toggle('offline', offline); }
function persist() { try { localStorage.setItem(STORE, JSON.stringify({references: state.demoReferences, history: state.demoHistory.slice(0, 50)})); } catch { notice('Browser storage is unavailable or full. This session still works, but demo history may not survive a reload.'); } }
function later(fn, ms) { const t = setTimeout(() => { state.timers.delete(t); fn(); }, ms); state.timers.add(t); return t; }
function cleanup() { state.source?.close(); state.source = null; state.controller?.abort(); state.controller = null; for (const timer of state.timers) clearTimeout(timer); state.timers.clear(); }
function validGeneration(g) { return g === state.generation; }
function route(name, id) { const value = API[name]; return typeof value === 'function' ? value(id) : value; }
async function request(name, options = {}, id) {
  const url = new URL(route(name, id), location.href);
  if (url.origin !== location.origin) throw new Error('API routes must use the same origin as this dashboard.');
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (options.signal?.aborted) controller.abort();
  options.signal?.addEventListener('abort', abort, {once: true});
  const timer = setTimeout(abort, 20000);
  try {
    const response = await fetch(url, {...options, signal: controller.signal, headers: {'Accept': 'application/json', ...(options.body ? {'Content-Type': 'application/json'} : {}), ...options.headers}});
    if (!response.ok) {
      let detail = ''; try { const data = await response.json(); detail = typeof data.detail === 'string' ? data.detail : ''; } catch { /* Keep HTTP error. */ }
      throw new Error(detail || `API request failed (${response.status} ${response.statusText}).`);
    }
    if (response.status === 204) return {};
    return await response.json();
  } finally { clearTimeout(timer); options.signal?.removeEventListener('abort', abort); }
}
function updateButtons() {
  $('inspect-button').disabled = state.busy || state.references.length === 0;
  $('reference-select').disabled = state.busy;
  $('verification-mode').disabled = state.busy;
  for (const button of document.querySelectorAll('[data-scenario]')) button.disabled = state.busy;
  $('calibrate-button').disabled = state.calibrationBusy || state.busy;
}
function showView(view) {
  if (!['inspect','history','references','calibration'].includes(view)) view = 'inspect';
  state.view = view;
  for (const section of document.querySelectorAll('.view')) section.hidden = section.id !== `view-${view}`;
  for (const link of document.querySelectorAll('[data-view]')) {
    const active = link.dataset.view === view; link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  }
  if (view === 'history') renderHistory();
}
function go(view) { if (location.hash === `#${view}`) showView(view); else location.hash = view; }
function setProgress(percent, message, stage = null) {
  const p = Math.max(0, Math.min(100, Number(percent) || 0));
  $('progress-track').setAttribute('aria-valuenow', String(p)); $('progress-fill').style.width = `${p}%`;
  $('progress-status').textContent = message;
  const current = stage ?? Math.min(3, Math.floor(p / 25));
  for (const step of document.querySelectorAll('[data-step]')) { const i = Number(step.dataset.step); step.classList.toggle('done', p === 100 || i < current); step.classList.toggle('current', p < 100 && i === current); }
}
function setVerdict(label, status, summary) {
  $('verdict-badge').style.fontSize = ''; $('verdict-label').textContent = label; $('verdict-badge').className = `verdict-badge ${status}`;
  $('verdict-badge').querySelector('.verdict-symbol').textContent = status === 'pass' ? '✓' : status === 'fail' ? '×' : status === 'busy' ? '◌' : '◎';
  $('verdict-summary').textContent = summary;
}
function resetResult() {
  state.result = null; state.selectedFinding = null;
  setVerdict('READY', 'idle', 'Your next circuit, under the lens.');
  for (const id of ['part-count','finding-count','inspection-time']) $(id).textContent = '—';
  $('findings-counter').textContent = '0';
  $('findings-list').replaceChildren(empty('Let’s connect the dots.', 'Run an inspection to locate faults on the board.'));
  $('board-image').src = '/static/board-base.svg'; $('board-image').alt = state.demo ? 'Simulated empty breadboard' : 'Placeholder breadboard; no live image received';
  $('image-source-label').textContent = state.demo ? 'SIMULATION' : 'AWAITING IMAGE';
  $('result-provenance').textContent = state.demo ? 'Demo results are simulated in your browser.' : 'Waiting for a live backend inspection.';
  $('board-hint').textContent = 'Run an inspection to trace the circuit';
  setProgress(0, 'Ready when you are'); renderOverlay();
}
function empty(title, message) { const box = element('div', undefined, 'empty-state'); box.append(element('strong', title), element('p', message)); return box; }
function renderReferences() {
  const selected = $('reference-select').value;
  $('reference-select').replaceChildren(); $('references-list').replaceChildren();
  for (const ref of state.references) {
    const option = element('option', ref.name); option.value = ref.id; $('reference-select').append(option);
    const row = element('div', undefined, 'reference-item'), text = element('div');
    text.append(element('h3', ref.name), element('p', `${ref.parts?.length ?? ref.part_count ?? '—'} components · ${state.demo ? 'Browser demo library' : 'Saved on server'}`));
    const use = element('button', 'Use reference →', 'text-button'); use.type = 'button';
    use.addEventListener('click', () => { if (state.busy) return notice('Wait for the current inspection to finish.'); $('reference-select').value = ref.id; go('inspect'); });
    row.append(text, use); $('references-list').append(row);
  }
  if (state.references.some(r => String(r.id) === selected)) $('reference-select').value = selected;
  if (!state.references.length) { $('reference-select').append(element('option', 'No references available')); $('references-list').append(empty('No saved references.', 'Create a reference to start inspecting.')); }
  $('references-count').textContent = String(state.references.length); updateButtons();
}
function dateLabel(value) { const d = new Date(value); return Number.isNaN(d.getTime()) ? 'Timestamp unavailable' : d.toLocaleString(undefined, {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit'}); }
function renderHistory() {
  $('history-body').replaceChildren(); $('history-empty').hidden = state.history.length > 0;
  for (const item of state.history) {
    const row = element('tr'), when = element('td'); when.append(element('span', dateLabel(item.timestamp || item.created_at)), element('small', String(item.id || '').slice(0, 26)));
    row.append(when, element('td', item.reference_name || item.reference?.name || 'Unnamed reference'), element('td', item.mode || 'connectivity'), element('td', item.simulation ? 'DEMO' : 'LIVE'));
    const result = element('td'), passed = item.verdict?.passed ?? item.passed;
    const label = passed === true ? 'PASS' : passed === false ? 'FAIL' : 'INCOMPLETE';
    result.append(element('span', label, `pill ${passed === true ? 'pass' : 'fail'}`)); row.append(result);
    const action = element('td'), button = element('button', 'View →', 'text-button');
    button.addEventListener('click', async () => {
      if (state.busy) return notice('Wait for the current inspection to finish.');
      const g = state.generation; button.disabled = true;
      try { const full = item.verdict?.snapped_parts ? item : await request('inspection', {}, item.id); if (!validGeneration(g)) return; displayResult(full, false); go('inspect'); }
      catch (e) { if (validGeneration(g)) notice(e.message); }
      finally { button.disabled = false; }
    }); action.append(button); row.append(action); $('history-body').append(row);
  }
}
function normalizeResult(raw) { return normalizeInspection(raw); }
function displayResult(raw, addHistory = true) {
  const data = normalizeResult(raw);
  data.simulation = data.simulation === true;
  data.id ??= state.requestId || `inspection-${Date.now()}`;
  data.timestamp ??= data.created_at || new Date().toISOString();
  data.mode ??= $('verification-mode').value;
  data.reference_name ??= state.references.find(r => String(r.id) === String(data.reference_id || $('reference-select').value))?.name || data.reference_id || 'Reference circuit';
  const elapsed = Number(data.elapsed_ms ?? performance.now()-state.start);
  data.elapsed_ms = Number.isFinite(elapsed) ? Math.max(0, elapsed) : 0;
  state.result = data; state.selectedFinding = null;
  if (['placement','connectivity'].includes(data.mode)) $('verification-mode').value = data.mode;
  if (state.references.some(r => String(r.id) === String(data.reference_id))) $('reference-select').value = data.reference_id;
  const lut = normalizeLUT(data.lut || data.hole_lut || state.calibration?.lut || createLUT());
  state.lut = lut;
  const rejected = !data.verdict || data.aborted || data.quality?.passed === false || data.quality?.failures?.length;
  const verdict = data.verdict, passed = !rejected && verdict.passed === true;
  setVerdict(data.status === 'error' ? 'ERROR' : rejected ? 'REJECTED' : passed ? 'PASS' : 'FAIL', passed ? 'pass' : 'fail', data.status === 'error' ? (data.error || 'Backend inspection failed; no circuit verdict.') : rejected ? 'Image quality was insufficient. Correct the capture conditions and try again.' : passed ? 'The circuit matches your reference. Every checked connection is accounted for.' : 'A connection needs your attention. Review the highlighted findings.');
  if (rejected) $('verdict-badge').style.fontSize = '27px'; else $('verdict-badge').style.fontSize = '';
  $('part-count').textContent = String(verdict?.snapped_parts?.length ?? 0);
  $('finding-count').textContent = String(verdict?.findings?.length ?? data.quality?.failures?.length ?? 0);
  $('inspection-time').textContent = `${(data.elapsed_ms/1000).toFixed(1)}s`;
  $('result-provenance').textContent = data.simulation ? 'SIMULATED · seeded browser scenario, not a camera result.' : `BACKEND · ${dateLabel(data.timestamp)}`;
  $('board-hint').textContent = data.lut || data.hole_lut || state.calibration?.lut ? 'Hover or select a finding to locate it' : data.simulation ? 'Hover or select a finding to locate it' : 'Default board grid · calibrated LUT not supplied';
  const url = data.rectified_image_url || data.image_url;
  let imageURL = null;
  if (url) {
    try { const u = new URL(url, location.href); if (u.origin === location.origin && ['http:','https:'].includes(u.protocol)) imageURL = u.href; else if (/^data:image\/(png|jpeg|webp);base64,/.test(url)) imageURL = url; } catch { /* Unsafe URL rejected. */ }
  }
  $('board-image').src = imageURL || '/static/board-base.svg';
  $('board-image').alt = imageURL ? 'Rectified inspection image' : data.simulation ? 'Simulated circuit board' : 'Placeholder board; backend did not return a rectified image';
  $('image-source-label').textContent = data.simulation ? 'SIMULATION' : imageURL ? 'BACKEND IMAGE' : 'IMAGE UNAVAILABLE';
  if (!data.simulation && !imageURL) notice('A verdict was received without a usable rectified image. The board below is a labeled placeholder.');
  setProgress(100, rejected ? 'Stopped at quality gate' : data.simulation ? 'Simulation complete' : 'Inspection complete');
  renderOverlay(); renderFindings();
  if (addHistory) {
    state.history = [data, ...state.history.filter(h => h.id !== data.id)].slice(0, 100);
    if (state.demo) { state.demoHistory = state.history.slice(0, 50); persist(); }
    renderHistory();
  }
}
function renderOverlay() {
  for (const id of ['grid-layer','component-layer','outline-layer','focus-layer']) $(id).replaceChildren();
  if ($('grid-toggle').checked) for (const point of Object.values(state.lut)) $('grid-layer').append(svg('circle', {cx:point.x, cy:point.y, r:2, fill:'#638b64', opacity:.4}));
  const parts = state.result?.verdict?.snapped_parts || [], findings = state.result?.verdict?.findings || [];
  for (const part of parts) {
    if (state.result.simulation) drawComponent(part);
    if (!Array.isArray(part.bbox) || part.bbox.length !== 4 || !part.bbox.every(Number.isFinite)) continue;
    const [x1,y1,x2,y2] = part.bbox;
    const outline = svg('rect', {x:x1-4, y:y1-4, width:Math.max(1,x2-x1+8), height:Math.max(1,y2-y1+8), rx:7, stroke:COLORS[partStatus(part, findings)], class:'part-outline', tabindex:0, role:'button', 'aria-label':`${part.type} ${part.id}: ${partStatus(part, findings)}`});
    outline.dataset.partId = part.id;
    const title = svg('title'); title.textContent = `${part.id} · ${part.type}`; outline.append(title);
    const select = () => { const index = findings.findIndex(f => f.part_id === part.id); if (index >= 0) selectFinding(index); else highlightPart(part.id); };
    outline.addEventListener('click', select); outline.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(); } });
    $('outline-layer').append(outline);
  }
  if (state.selectedFinding !== null) focusFinding(state.selectedFinding);
}
function drawComponent(part) {
  const points = Object.values(part.pins || {}).map(h => state.lut[h]); if (points.length < 2 || points.some(p => !p)) return;
  const [a,b] = points, group = svg('g'), stroke = part.id === 'W_GND' ? '#2c3430' : part.id === 'W_SHORT' ? '#cc4d38' : '#317d90';
  if (part.type === 'wire') group.append(svg('path', {d:`M${a.x} ${a.y} Q${(a.x+b.x)/2+12} ${(a.y+b.y)/2-10} ${b.x} ${b.y}`, fill:'none', stroke, 'stroke-width':7, 'stroke-linecap':'round'}));
  else {
    group.append(svg('line', {x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#66716b','stroke-width':4}));
    const cx=(a.x+b.x)/2,cy=(a.y+b.y)/2,angle=Math.atan2(b.y-a.y,b.x-a.x)*180/Math.PI;
    if (part.type === 'resistor') {
      const body=svg('g',{transform:`translate(${cx} ${cy}) rotate(${angle})`}); body.append(svg('rect',{x:-25,y:-10,width:50,height:20,rx:6,fill:'#c9b48a',stroke:'#877754','stroke-width':2}));
      for (const [i,c] of ['#8b452a','#2c3026','#a14f31','#b4a543'].entries()) body.append(svg('rect',{x:-17+i*10,y:-9,width:4,height:18,fill:c})); group.append(body);
    } else {
      group.append(svg('circle',{cx,cy,r:12,fill:part.id === 'LED1' ? '#dc5641' : '#8fbd6b',stroke:'#8a2d24','stroke-width':2}),svg('circle',{cx:cx-4,cy:cy-4,r:4,fill:'#f8c4a8',opacity:.7}));
    }
  }
  for(const p of points) group.append(svg('circle',{cx:p.x,cy:p.y,r:3,fill:'#494f48'}));
  $('component-layer').append(group);
}
function clearHighlights() { $('focus-layer').replaceChildren(); for (const node of $('outline-layer').children) node.classList.remove('highlighted'); }
function highlightPart(id) { clearHighlights(); for (const node of $('outline-layer').children) if (node.dataset.partId === id) node.classList.add('highlighted'); }
function focusFinding(index) {
  const finding = state.result?.verdict?.findings?.[index]; if (!finding) return;
  clearHighlights();
  const parts = state.result.verdict.snapped_parts || [];
  for (const node of $('outline-layer').children) if (!finding.part_id || node.dataset.partId === finding.part_id) node.classList.add('highlighted');
  const part = parts.find(p => p.id === finding.part_id);
  const holes = finding.holes?.length ? finding.holes : Object.values(part?.pins || {});
  for (const hole of holes) markHole(hole, false);
  for (const [i, hole] of (finding.expected_holes || []).entries()) {
    markHole(hole, true);
    const end = state.lut[hole], start = state.lut[holes[i] || holes[0]];
    if (end && start) $('focus-layer').append(svg('line',{x1:start.x,y1:start.y,x2:end.x,y2:end.y,class:'ghost'}));
  }
  // Also draw an expected ghost body when both expected pins are supplied.
  const expected = (finding.expected_holes || []).map(h=>state.lut[h]).filter(Boolean);
  if(expected.length>=2) $('focus-layer').append(svg('line',{x1:expected[0].x,y1:expected[0].y,x2:expected[1].x,y2:expected[1].y,class:'ghost'}));
}
function markHole(hole, expected) {
  const p=state.lut[hole]; if(!p) return;
  $('focus-layer').append(svg('circle',{cx:p.x,cy:p.y,r:8,class:expected?'ghost':'focus-hole'}));
  const label=svg('text',{x:p.x+10,y:p.y-10,fill:'#702e09','font-size':13,'font-family':'monospace','font-weight':600,'paint-order':'stroke',stroke:'#f4e9d4','stroke-width':3}); label.textContent=hole; $('focus-layer').append(label);
}
function selectFinding(index) {
  state.selectedFinding = state.selectedFinding === index ? null : index;
  for (const button of document.querySelectorAll('.finding')) { const selected = Number(button.dataset.index) === state.selectedFinding; button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected)); }
  if(state.selectedFinding===null) clearHighlights(); else focusFinding(state.selectedFinding);
}
function renderFindings() {
  const findings=state.result?.verdict?.findings||[], list=$('findings-list'); list.replaceChildren(); $('findings-counter').textContent=String(findings.length);
  if(!findings.length){ const rejected=state.result?.quality?.failures; const noVerdict = !state.result?.verdict; list.append(empty(noVerdict ? 'No circuit verdict.' : rejected?.length ? 'Image needs another look.' : 'All checked connections match.', rejected?.length ? `Failed checks: ${rejected.join(', ')}. Adjust the camera or lighting and retry.` : noVerdict ? (state.result?.error || 'The inspection did not complete.') : 'No circuit faults were reported for this inspection.')); return; }
  const titles={POWER_SHORT:'Power is connected to ground',REVERSED_POLARITY:'LED polarity is reversed',MISPLACED_PIN:'A pin is in the wrong hole',OPEN_CONNECTION:'A connection is open',UNCERTAIN:'A pin could not be located reliably',MISSING_PART:'A component is missing',EXTRA_PART:'An extra component was found',UNEXPECTED_CONNECTION:'An unexpected connection was found'};
  findings.forEach((finding,index)=>{
    const kind=finding.code==='UNCERTAIN'?'uncertain':finding.severity==='critical'?'critical':'warning';
    const button=element('button',undefined,`finding ${kind}`); button.dataset.index=String(index); button.setAttribute('aria-pressed','false');
    const top=element('span',undefined,'finding-top'); top.append(element('span',finding.code?.replaceAll('_',' ')||'FINDING'),element('span',finding.part_id||'CIRCUIT'));
    button.append(top,element('strong',titles[finding.code]||'Review this connection'),element('p',finding.message||'Review the highlighted component.'));
    button.addEventListener('mouseenter',()=>focusFinding(index)); button.addEventListener('focus',()=>focusFinding(index));
    const restore=()=>{ if(state.selectedFinding===null) clearHighlights(); else focusFinding(state.selectedFinding); };
    button.addEventListener('mouseleave',restore); button.addEventListener('blur',restore); button.addEventListener('click',()=>selectFinding(index)); list.append(button);
  });
}
async function inspect() {
  if(state.busy) return;
  const ref=state.references.find(r=>String(r.id)===$('reference-select').value);
  if(!ref) return notice('Select a saved reference first.');
  if(state.demo && ref.id!==SEED_REFERENCE.id) return notice('Seeded scenarios use the bundled LED + resistor reference. Select it to simulate, or use live mode to verify your own circuit.');
  cleanup(); const generation=++state.generation; notice(); state.busy=true; state.start=performance.now(); state.requestId=null; resetResult(); updateButtons();
  setVerdict('INSPECTING','busy',state.demo?'Running a seeded browser scenario…':'Waiting for the camera and inspection service…');
  setProgress(2,state.demo?'Starting simulation':'Requesting capture');
  const mode=$('verification-mode').value;
  if(state.demo){
    const messages=['Checking simulated capture','Locating simulated components','Reconstructing seeded connections','Comparing seeded circuit'];
    messages.forEach((message,i)=>later(()=>{if(validGeneration(generation))setProgress(10+i*25,message,i);},i*120));
    later(()=>{if(!validGeneration(generation))return; const result=makeDemoResult(state.scenario,mode); result.elapsed_ms=performance.now()-state.start; displayResult(result); state.busy=false; updateButtons();},520);
    return;
  }
  state.controller=new AbortController();
  try {
    const response=await request('inspect',{method:'POST',body:JSON.stringify({reference_id:ref.id,mode}),signal:state.controller.signal});
    if(!validGeneration(generation))return;
    if(response.verdict || response.result?.verdict){ complete(response,generation); return; }
    const id=response.id??response.inspection_id;
    if(id===undefined||id===null)throw new Error('Backend did not return an inspection id.');
    state.requestId=String(id); listenEvents(state.requestId,generation);
  } catch(e){if(validGeneration(generation))failInspection(e.message);}
}
function complete(data,generation) {
  if(!validGeneration(generation)||!state.busy)return;
  try{displayResult(data);cleanup();state.busy=false;updateButtons();setConnection('Live API · inspection complete');}
  catch(e){failInspection(e.message);}
}
function failInspection(message) {
  cleanup();state.busy=false;updateButtons();setVerdict('ERROR','fail','No completed circuit verdict. Retry after resolving the issue.');
  $('result-provenance').textContent = state.demo ? 'Simulation interrupted; no verdict.' : 'BACKEND ERROR · no completed circuit verdict.';
  $('progress-status').textContent='Inspection interrupted';notice(message);setConnection(state.demo?'Browser simulation':'Live API · inspection interrupted',true);
}
function listenEvents(id,generation) {
  let errors=0,finishing=false;
  const eventURL=new URL(route('events',id),location.href);
  if(eventURL.origin!==location.origin)throw new Error('SSE events must use the dashboard origin.');
  state.source=new EventSource(eventURL);
  const finish=async data=>{
    if(finishing||!validGeneration(generation))return; finishing=true;
    try{const result=data.verdict||data.result?.verdict||data.aborted||data.quality?.failures?.length?data:await request('inspection',{},id);complete(result,generation);}
    catch(e){if(validGeneration(generation))failInspection(e.message);}
  };
  const receive=event=>{
    if(!validGeneration(generation))return;
    let data;try{data=JSON.parse(event.data);}catch{failInspection('The inspection event stream returned invalid JSON.');return;}
    errors=0;
    if(data.verdict||data.result?.verdict||data.aborted||['complete','completed','done','result'].includes(data.status||data.type)||['complete','completed','done','result','aborted'].includes(event.type)){finish(data);return;}
    if(['error','failed'].includes(data.status||data.type)||event.type==='error'){failInspection(data.message||data.error||'The inspection service reported an error.');return;}
    const progress=eventProgress(data);
    setProgress(progress.percent,progress.message,progress.stage);
  };
  state.source.onmessage=receive;
  for(const type of ['progress','stage','capturing','registering','detecting','checking','complete','completed','done','result','aborted'])state.source.addEventListener(type,receive);
  state.source.addEventListener('error',event=>{
    if(event.data){receive(event);return;}
    if(!validGeneration(generation)||finishing)return;
    errors++;$('progress-status').textContent='Event stream disconnected · reconnecting';
    if(errors>=3||state.source?.readyState===EventSource.CLOSED)finish({});
  });
  later(()=>{if(validGeneration(generation)&&state.busy)failInspection('Inspection timed out after two minutes. No completed verdict was received.');},120000);
}
async function loadLive(generation=state.generation) {
  setConnection('Connecting to live API…',true);
  const results=await Promise.allSettled([request('references'),request('history'),request('calibration')]);
  if(!validGeneration(generation)||state.demo)return;
  const [refs,history,calibration]=results;
  if(refs.status==='fulfilled'){
    const rows=refs.value.references||refs.value;
    state.references=Array.isArray(rows)?rows.map(normalizeReference).filter(r=>r?.id!==undefined&&typeof r.name==='string'):[];
    setConnection('Live API connected');
  }else{state.references=[];setConnection('Live API unavailable · demo remains available',true);notice(`Cannot load live references: ${refs.reason.message}`);}
  if(history.status==='fulfilled')state.history=normalizeHistory(history.value,state.references);
  if(calibration.status==='fulfilled'){state.calibration=normalizeCalibration(calibration.value);state.lut=normalizeLUT(state.calibration.lut||state.calibration.hole_lut);}
  renderReferences();renderHistory();renderCalibration();renderOverlay();
  const missing=[history.status==='rejected'?'history':null,calibration.status==='rejected'?'calibration':null].filter(Boolean);
  if(refs.status==='fulfilled'&&missing.length)notice(`Live references loaded, but ${missing.join(' and ')} could not be loaded. Use Refresh or retry calibration.`);
}
function renderCalibration() {
  const data=state.calibration, ready=data?.calibrated===true||data?.status==='calibrated';
  $('calibration-badge').textContent=ready?(state.demo?'SIMULATED GRID':'CALIBRATED'):data?.registered?'REGISTERED / LUT PENDING':'NOT CALIBRATED';
  $('lut-count').textContent=ready?String(data.hole_count??Object.keys(normalizeLUT(data.lut||data.hole_lut)).length):data?.registered?'Not supplied by backend':'—';
  $('calibration-date').textContent=data?.updated_at?dateLabel(data.updated_at):'—';
  $('calibration-source').textContent=state.demo?'Browser simulation':'Backend (camera or synthetic input)';
  $('calibration-message').textContent=state.demo?'Demo calibration illustrates the hole grid; it does not calibrate a camera.':ready?'Calibration is ready for inspection.':data?.registered?'Board markers registered. The backend has not supplied a calibrated hole LUT.':'Calibrate the empty board before using live detections.';
}
async function calibrate() {
  if(state.calibrationBusy||state.busy)return;
  const g=state.generation;state.calibrationBusy=true;updateButtons();notice();$('calibration-message').textContent='Calibrating…';
  try{
    const data=state.demo?{calibrated:true,lut:createLUT(),updated_at:new Date().toISOString()}:await request('calibrate',{method:'POST',body:'{}'});
    if(!validGeneration(g))return;
    state.calibration=normalizeCalibration(data);state.lut=normalizeLUT(state.calibration.lut||state.calibration.hole_lut);renderCalibration();renderOverlay();
    if(!state.calibration.calibrated&&!state.calibration.registered)$('calibration-message').textContent=data.message||'Calibration requested. Refresh live mode after the backend finishes.';
  }catch(e){if(validGeneration(g)){$('calibration-message').textContent='Calibration could not be completed.';notice(e.message);}}
  finally{if(validGeneration(g)){state.calibrationBusy=false;updateButtons();}}
}
async function saveReference(event) {
  event.preventDefault();const button=event.currentTarget.querySelector('[type="submit"]'),g=state.generation;
  $('reference-error').textContent='';button.disabled=true;
  try{
    if(new Blob([$('reference-json').value]).size>262144)throw new Error('Reference JSON must be 256 KB or smaller.');
    const definition=JSON.parse($('reference-json').value);definition.name=$('reference-name').value.trim();validateReference(definition);
    let savedRef;
    if(state.demo){savedRef={...definition,id:`local-${Date.now()}-${Math.random().toString(36).slice(2,7)}`};state.demoReferences.push(savedRef);persist();}
    else{const response=await request('references',{method:'POST',body:JSON.stringify(definition)});savedRef=normalizeReference(response.reference||response);if(savedRef.id===undefined)throw new Error('Reference saved but response omitted its id. Refresh the reference list.');}
    if(!validGeneration(g))return;
    state.references.push(savedRef);renderReferences();notice(state.demo?'Reference saved in this browser. Seeded simulations still use the bundled LED circuit.':'Reference saved to the backend.');
  }catch(e){if(validGeneration(g))$('reference-error').textContent=e instanceof SyntaxError?'The circuit definition is not valid JSON.':e.message;}
  finally{button.disabled=false;}
}
function switchMode() {
  cleanup();state.generation++;state.busy=false;state.calibrationBusy=false;state.demo=$('demo-toggle').checked;state.calibration=null;state.history=[];state.lut=state.demo?createLUT():{};notice();
  $('demo-scenarios').hidden=!state.demo;$('footer-mode').textContent=state.demo?'SIMULATION WORKSPACE':'LIVE CAMERA WORKSPACE';
  document.querySelector('.mode-switch small').textContent=state.demo?'SIMULATED INPUT':'LIVE API INPUT';
  state.references=state.demo?[structuredClone(SEED_REFERENCE),...state.demoReferences]:[];
  state.history=state.demo?[...state.demoHistory]:[];resetResult();renderReferences();renderHistory();renderCalibration();updateButtons();
  if(state.demo)setConnection('Browser simulation · no camera required');else loadLive();
}
$('inspect-button').addEventListener('click',inspect);
$('demo-toggle').addEventListener('change',switchMode);
$('grid-toggle').addEventListener('change',renderOverlay);
$('clear-focus').addEventListener('click',()=>{state.selectedFinding=null;clearHighlights();for(const button of document.querySelectorAll('.finding')){button.classList.remove('selected');button.setAttribute('aria-pressed','false');}});
$('calibrate-button').addEventListener('click',calibrate);
$('reference-form').addEventListener('submit',saveReference);
$('reference-file').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>262144)throw new Error('JSON file must be 256 KB or smaller.');const value=JSON.parse(await file.text());validateReference(value);$('reference-json').value=JSON.stringify(value,null,2);$('reference-name').value=value.name;$('reference-error').textContent='';}catch(e){$('reference-error').textContent=e.message;}});
$('refresh-history').addEventListener('click',async()=>{if(state.demo){state.history=[...state.demoHistory];renderHistory();return;}const g=state.generation;try{const data=await request('history');if(validGeneration(g)){state.history=normalizeHistory(data,state.references);renderHistory();}}catch(e){if(validGeneration(g))notice(e.message);}});
for(const button of document.querySelectorAll('[data-scenario]'))button.addEventListener('click',()=>{if(state.busy)return;state.scenario=button.dataset.scenario;for(const other of document.querySelectorAll('[data-scenario]')){const active=other===button;other.classList.toggle('active',active);other.setAttribute('aria-pressed',String(active));}inspect();});
document.addEventListener('keydown', e => { if (e.key === 'Enter' && state.view === 'inspect' && !e.target.closest('input,select,textarea,button,a,[contenteditable]')) { e.preventDefault(); inspect(); } });
window.addEventListener('hashchange',()=>showView(location.hash.slice(1)));
window.addEventListener('beforeunload',cleanup);
$('board-image').addEventListener('error',()=>{if($('board-image').getAttribute('src')!=='/static/board-base.svg'){$('board-image').src='/static/board-base.svg';$('image-source-label').textContent='IMAGE UNAVAILABLE';notice('The rectified image could not be loaded. Showing a placeholder board.');}});
const template=structuredClone(SEED_REFERENCE);delete template.id;$('reference-json').value=JSON.stringify(template,null,2);
switchMode();showView(location.hash.slice(1)||'inspect');
