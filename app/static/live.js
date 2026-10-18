import {escapeHTML as esc, presentVerdict, makeReference} from './live-state.mjs';

const $ = id => document.getElementById(id);
const state = {backend:readPreference('sherlock.live.backend'), data:null, page:'inspect', source:'camera', stream:null, running:false, busy:false, session:null, sequence:0, generation:0, result:null, capturedAt:0, photo:null, calibration:null, anchors:{}, setupId:readPreference('sherlock.live.setup'), referenceId:readPreference('sherlock.live.reference')};
const titles = {inspect:'Live wiring', setup:'Board setup', references:'Reference circuits', history:'Inspection history'};
let loopTimer = null;
let noticeTimer = null;

function readPreference(key) {try {return localStorage.getItem(key) || '';} catch {return '';}}
function savePreference(key, value) {try {localStorage.setItem(key, value);} catch { /* Storage-disabled browsers still work for this visit. */ }}
function icon(name) {return `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;}
function notice(message, success = false) {
  clearTimeout(noticeTimer); $('notice').textContent = message; $('notice').classList.toggle('success', success); $('notice').hidden = false;
  if (success) noticeTimer = setTimeout(() => { $('notice').hidden = true; }, 6500);
}
function errorText(error) {return error?.name === 'TimeoutError' ? 'The backend took too long. Try again; the first model load can take a moment.' : error.message || String(error);}
async function api(path, {method = 'GET', body, keepalive = false} = {}) {
  const response = await fetch(`${state.backend}/api/live${path}`, {method, headers:body ? {'Content-Type':'application/json'} : {}, ...(body ? {body:JSON.stringify(body)} : {}), signal:keepalive ? undefined : AbortSignal.timeout(45000), keepalive});
  let data; try {data = await response.json();} catch {throw new Error('The backend did not return a valid API response. Check the connection address.');}
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(item => `${item.loc?.slice(1).join('.') || 'Input'}: ${item.msg}`).join('; ') : data.detail;
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}).`);
  }
  return data;
}
function setBusy(value, loading = false) {state.busy = value; $('stage-loading').hidden = !loading; $('connection-settings').disabled = value; $('setup-columns').disabled = value; $('setup-name').disabled = value; $('reset-anchors').disabled = value || !Object.keys(state.anchors).length; updateControls();}
function updateControls() {
  const camera = Boolean(state.stream);
  $('camera-button').textContent = ''; $('camera-button').insertAdjacentHTML('beforeend', icon('camera')); $('camera-button').append(document.createTextNode(camera ? 'Disconnect camera' : 'Connect camera'));
  $('camera-button').disabled = state.busy && !state.running;
  $('take-photo-button').disabled = !camera || state.busy || state.running;
  $('upload-button').disabled = state.busy || state.running;
  $('empty-upload').disabled = state.busy;
  $('empty-camera').disabled = state.busy;
  $('live-button').disabled = !state.running && (state.busy || !camera || !state.setupId || !state.referenceId || !state.data?.model?.available);
  $('live-button').innerHTML = state.running ? 'Stop live check <span aria-hidden="true">■</span>' : `Start live check ${icon('arrow')}`;
  $('live-button').classList.toggle('secondary', state.running);
  $('camera-indicator').classList.toggle('on', camera);
  $('feed-dot').classList.toggle('on', state.running);
  $('feed-label').textContent = state.running ? 'LIVE · FRESH FRAMES' : state.source === 'photo' && state.result ? 'PHOTO INSPECTION' : camera ? 'CAMERA CONNECTED' : 'AWAITING INPUT';
  $('capture-copy').textContent = state.running ? 'Watching your connections. Keep all four markers visible.' : camera ? !state.setupId ? 'Camera connected. Take a photo to detect wire and resistor endpoints.' : !state.referenceId ? 'Take a photo for endpoint detection, or choose a reference for wiring checks.' : 'Camera connected. Take a photo or start a live check.' : 'Camera is off. Your workspace is ready.';
  $('reference-select').disabled = state.running || state.busy; $('setup-select').disabled = state.running || state.busy;
}
async function refreshStatus() {
  const backend = state.backend;
  try {
    const data = await api('/status');
    if (state.backend !== backend) return;
    state.data = data;
    const model = state.data.model;
    $('connection-text').textContent = model.available ? model.loaded ? 'Local model connected' : 'Trained model available' : 'Model weights missing';
    $('connection-pill').classList.toggle('offline', !model.available);
    $('lab-location').textContent = state.backend ? 'CONNECTED API' : 'LOCAL WORKSPACE';
    populateSelect('reference-select', state.data.references, state.referenceId, 'Choose a reference circuit');
    populateSelect('setup-select', state.data.setups, state.setupId, 'No board calibrated');
    state.referenceId = $('reference-select').value; state.setupId = $('setup-select').value;
    renderReferences(); updateControls();
  } catch (error) {
    if (state.backend !== backend) return;
    state.data = null; $('connection-pill').classList.add('offline'); $('connection-text').textContent = 'Backend disconnected';
    $('reference-select').innerHTML = '<option value="">Backend unavailable</option>';
    $('setup-select').innerHTML = '<option value="">Backend unavailable</option>';
    updateControls(); notice(`Couldn’t reach the inspection backend. ${errorText(error)}`);
  }
}
function populateSelect(id, items = [], selected, empty) {
  const select = $(id); select.replaceChildren(new Option(empty, ''));
  for (const item of items) select.add(new Option(item.name, item.id));
  select.value = items.some(item => item.id === selected) ? selected : items[0]?.id || '';
}
function renderReferences() {
  const references = state.data?.references || [];
  $('reference-library').innerHTML = references.length ? references.map(ref => `<button class="reference-option" type="button" data-reference="${esc(ref.id)}"><strong>${esc(ref.name)}</strong><small>${ref.parts.length} COMPONENT${ref.parts.length === 1 ? '' : 'S'} · USE THIS CIRCUIT ↗</small></button>`).join('') : '<p class="quiet-empty">Your first circuit starts here.<br>Save the one-wire starter on the right.</p>';
}
function navigate() {
  const page = location.hash.slice(1); const next = titles[page] ? page : 'inspect';
  if (next !== 'inspect') {stopLive(); disconnectCamera();}
  state.page = next;
  document.querySelectorAll('[data-page]').forEach(element => {element.hidden = element.dataset.page !== next;});
  document.querySelectorAll('[data-view]').forEach(element => {const active = element.dataset.view === next; element.classList.toggle('active', active); if (active) element.setAttribute('aria-current', 'page'); else element.removeAttribute('aria-current');});
  $('breadcrumb-current').textContent = titles[next];
  if (next === 'history') loadHistory();
}
function setSource(source) {
  state.source = source; $('camera-tab').classList.toggle('selected', source === 'camera'); $('photo-tab').classList.toggle('selected', source === 'photo');
}
async function connectCamera() {
  if (state.busy) return;
  if (state.stream) {disconnectCamera(); return;}
  if (!navigator.mediaDevices?.getUserMedia) {notice('Camera access needs localhost or HTTPS. You can also upload a board photo.'); return;}
  setBusy(true);
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}, width:{ideal:1920}, height:{ideal:1080}}, audio:false});
    if (state.page !== 'inspect') {stream.getTracks().forEach(track => track.stop()); return;}
    state.stream = stream; $('camera-video').srcObject = stream; await $('camera-video').play();
    for (const track of stream.getVideoTracks()) track.addEventListener('ended', () => {if (state.stream === stream) disconnectCamera();});
    setSource('camera'); state.result = null; showRawCamera(); renderResult();
  } catch (error) {
    if (state.stream) disconnectCamera();
    notice(error.name === 'NotAllowedError' ? 'Camera permission was declined. Enable it in your browser or upload a photo.' : error.name === 'NotFoundError' ? 'No camera was found. Connect a webcam or upload a photo.' : `Couldn’t open the camera. ${errorText(error)}`);
  } finally {setBusy(false);}
}
function showRawCamera() {
  $('board-result').hidden = true; $('empty-state').hidden = Boolean(state.stream); $('camera-video').hidden = !state.stream;
  $('stage-source-label').hidden = !state.stream; $('stage-source-label').textContent = 'CAMERA PREVIEW · NO CURRENT VERDICT';
  if (state.stream) $('image-dimensions').textContent = `${$('camera-video').videoWidth} × ${$('camera-video').videoHeight} PX`;
}
function disconnectCamera() {
  stopLive(); const stream = state.stream; state.stream = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  $('camera-video').srcObject = null; $('camera-video').hidden = true;
  if (state.source === 'camera') {state.result = null; $('board-result').hidden = true; $('empty-state').hidden = false; $('stage-source-label').hidden = true; $('image-dimensions').textContent = 'NO FRAME YET'; renderResult();}
  updateControls();
}
function stopLive() {
  state.running = false; state.generation++; clearTimeout(loopTimer);
  const session = state.session; state.session = null;
  if (session) api(`/sessions/${encodeURIComponent(session)}`, {method:'DELETE', keepalive:true}).catch(() => {});
  if (state.source === 'camera' && state.result) {state.result = {status:'UNCERTAIN', message:'Live checking is paused. Start a new check for current feedback.', parts:[], detections:[]}; showRawCamera(); renderResult();}
  updateControls();
}
async function startLive() {
  if (state.running) {stopLive(); return;}
  if (state.busy || !state.stream || !state.setupId || !state.referenceId) return;
  setBusy(true); state.result = null; renderResult();
  try {
    const generation = ++state.generation;
    const session = await api('/sessions', {method:'POST', body:{setup_id:state.setupId, reference_id:state.referenceId}});
    if (generation !== state.generation || !state.stream || document.hidden) {api(`/sessions/${encodeURIComponent(session.session_id)}`, {method:'DELETE'}).catch(() => {}); return;}
    state.session = session.session_id; state.sequence = 0; state.running = true; setSource('camera');
    setBusy(false); liveFrame(generation);
  } catch (error) {notice(errorText(error));} finally {if (!state.running) setBusy(false);}
}
async function liveFrame(generation) {
  if (!state.running || state.generation !== generation || document.hidden) return;
  const capturedAt = performance.now();
  try {
    const video = $('camera-video');
    if (!video.videoWidth || video.readyState < 2) throw new Error('Waiting for a usable camera frame.');
    const canvas = $('capture-canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0); const image = canvas.toDataURL('image/jpeg', .9).split(',')[1];
    const result = await api(`/sessions/${encodeURIComponent(state.session)}/frame`, {method:'POST', body:{image_base64:image, mode:'live', frame_id:++state.sequence}});
    if (!state.running || state.generation !== generation) return;
    state.result = result; state.capturedAt = capturedAt; renderResult();
    if (performance.now() - capturedAt <= 2000) showResultImage(result, 'LIVE INSPECTION · VISIBLE CONNECTIVITY'); else showRawCamera();
    updateControls(); loopTimer = setTimeout(() => liveFrame(generation), 180);
  } catch (error) {
    if (state.generation !== generation) return;
    stopLive(); setBusy(false); notice(`Live checking stopped. ${errorText(error)}`);
  }
}
function showResultImage(result, label) {
  if (!result.image_base64) return;
  $('board-result').src = `data:image/jpeg;base64,${result.image_base64}`;
  $('board-result').hidden = false; $('camera-video').hidden = true; $('empty-state').hidden = true;
  $('stage-source-label').textContent = label; $('stage-source-label').hidden = false;
  $('image-dimensions').textContent = `${result.width} × ${result.height} PX`;
}
function renderResult() {
  const result = state.result; const presentation = presentVerdict(result, {live:state.running, now:performance.now(), capturedAt:state.capturedAt});
  $('verdict-panel').dataset.tone = presentation.tone;
  $('verdict-title').textContent = presentation.title; $('verdict-symbol').textContent = presentation.symbol;
  $('verdict-message').textContent = presentation.message; $('verdict-bottom').textContent = presentation.label;
  $('result-kind').textContent = state.running ? 'LIVE CHECK' : result?.mode === 'still' ? 'PHOTO CHECK' : result ? 'ENDPOINT PREVIEW' : 'NO INPUT';
  $('stability').hidden = !state.running || !result || presentation.expired;
  const frames = result?.stable_frames || 0, required = result?.required_stable_frames || 3;
  $('stability-count').textContent = `${Math.min(frames, required)} / ${required}`;
  $('stability-fill').style.width = `${Math.min(100, frames / required * 100)}%`;
  const quality = result?.quality; const grid = result?.grid_observation;
  setStep('track', quality?.passed === true && !presentation.expired, quality?.passed === false);
  const parts = result?.parts || [], detections = result?.detections || [];
  setStep('holes', Boolean(grid?.passed) && !presentation.expired, grid?.passed === false);
  setStep('match', presentation.tone === 'green', presentation.tone === 'red');
  const components = parts.length ? parts : detections;
  $('component-count').textContent = String(components.length).padStart(2,'0');
  $('component-list').innerHTML = components.length && !presentation.expired ? components.map((part, index) => {
    const pins = Object.values(part.pins || {}).map(pin => pin || '?');
    const location = pins.length ? pins.join(' ↔ ') : 'Hole IDs need calibration';
    return `<div class="component-row"><span class="component-icon">${icon('circuit')}</span><div><strong>${esc(part.type === 'wire' ? `Wire ${index + 1}` : `Resistor ${index + 1}`)}</strong><small>${esc(location)}</small></div><span class="confidence">${Math.round(Number(part.conf || 0)*100)}%<span class="sr-only"> model score</span></span></div>`;
  }).join('') : '<p class="quiet-empty">Connections will appear here.<br>One wire at a time.</p>';
  if (presentation.expired && state.stream) showRawCamera();
}
function setStep(name, passed, failed) {const row = $(`check-${name}`); row.classList.toggle('done', passed); row.classList.toggle('failed', failed); row.querySelector('.step-state').textContent = passed ? '✓' : failed ? '!' : '—';}
async function readImage(file) {
  if (!file || !['image/jpeg','image/png','image/webp'].includes(file.type)) throw new Error('Choose a JPEG, PNG, or WebP photo.');
  if (file.size > 8*1024*1024) throw new Error('Choose a photo smaller than 8 MB.');
  return new Promise((resolve,reject) => {const reader = new FileReader(); reader.onerror = () => reject(new Error('The photo could not be read.')); reader.onload = () => resolve(String(reader.result).split(',')[1]); reader.readAsDataURL(file);});
}
function takePhoto() {
  if (!state.stream || state.busy || state.running) return;
  try {
    const video = $('camera-video');
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) throw new Error('The camera is still starting. Wait for the preview, then take a photo.');
    const canvas = $('capture-canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const encoded = canvas.toDataURL('image/jpeg', .9).split(',')[1];
    const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
    // Capture the video pixels before inspectPhoto stops the stream. Both upload
    // and camera snapshots use the same real inference and still-image policy.
    void inspectPhoto(new File([bytes], 'breadboard-camera.jpg', {type:'image/jpeg'}));
  } catch (error) {notice(`Couldn’t take a photo. ${errorText(error)}`);}
}
async function inspectPhoto(file) {
  if (!file || state.busy) return;
  stopLive(); disconnectCamera(); setSource('photo'); state.result = null; renderResult(); setBusy(true,true);
  let photoSession = null;
  try {
    const image = await readImage(file); state.photo = image;
    let result;
    if (state.setupId && state.referenceId) {
      const session = await api('/sessions', {method:'POST', body:{setup_id:state.setupId, reference_id:state.referenceId}}); photoSession = session.session_id;
      result = await api(`/sessions/${encodeURIComponent(photoSession)}/frame`, {method:'POST', body:{image_base64:image, mode:'still', frame_id:1}});
    } else result = await api('/preview', {method:'POST', body:{image_base64:image}});
    state.result = result; state.capturedAt = performance.now(); showResultImage(result, result.mode === 'still' ? 'STILL PHOTO · LIVE CONFIRMATION REQUIRED' : 'ENDPOINT PREVIEW · NO CIRCUIT VERDICT'); renderResult();
    $('connection-text').textContent = 'Local model connected';
  } catch (error) {notice(errorText(error));} finally {if (photoSession) api(`/sessions/${encodeURIComponent(photoSession)}`, {method:'DELETE'}).catch(() => {}); setBusy(false); $('photo-input').value = '';}
}
function anchorNames() {const n = Number($('setup-columns').value); return ['a1',`a${n}`,`j${n}`,'j1'];}
function renderAnchors() {
  const names = anchorNames(), count = Object.keys(state.anchors).length;
  $('anchor-list').innerHTML = names.map((name,index) => `<li class="${state.anchors[name] ? 'complete' : index === count && state.calibration ? 'current' : ''}"><span>${index+1}. ${esc(name)}</span><span>${state.anchors[name] ? '✓' : '○'}</span></li>`).join('');
  $('save-setup').disabled = count !== 4 || !state.calibration;
  $('reset-anchors').disabled = !count;
  if (state.calibration) $('calibration-instructions').textContent = count < 4 ? `Click the center of ${names[count]} on your board. Use its printed labels to verify orientation.` : 'Four anchors selected. Check the labels, then save your calibration.';
}
async function prepareCalibration(file) {
  if (!file || state.busy) return;
  setBusy(true); $('calibration-upload').disabled = true; $('calibration-state').textContent = 'FINDING MARKERS';
  state.calibration = null; state.anchors = {}; renderAnchors();
  try {
    const image = await readImage(file); const columns = Number($('setup-columns').value);
    const result = await api('/calibration/preview', {method:'POST', body:{image_base64:image, columns}});
    const bitmap = new Image(); bitmap.src = `data:image/jpeg;base64,${result.image_base64}`; await bitmap.decode();
    if (Number($('setup-columns').value) !== columns) throw new Error('Board size changed. Choose the calibration photo again.');
    state.calibration = {image, bitmap, columns};
    const canvas = $('calibration-canvas'); canvas.width = result.width; canvas.height = result.height; canvas.hidden = false;
    $('calibration-placeholder').hidden = true; drawCalibration(); renderAnchors();
    $('calibration-state').textContent = 'MARK FOUR HOLES'; $('calibration-upload-label').textContent = file.name;
  } catch (error) {$('calibration-state').textContent = 'PHOTO NEEDS ATTENTION'; notice(errorText(error));} finally {$('calibration-upload').disabled = false; $('calibration-input').value = ''; setBusy(false);}
}
function drawCalibration() {
  if (!state.calibration) return;
  const canvas = $('calibration-canvas'), context = canvas.getContext('2d'); context.drawImage(state.calibration.bitmap,0,0,canvas.width,canvas.height);
  for (const [name,[x,y]] of Object.entries(state.anchors)) {context.strokeStyle = '#35502b'; context.fillStyle = '#d7f698'; context.lineWidth = 2; context.beginPath(); context.arc(x,y,7,0,Math.PI*2); context.fill(); context.stroke(); context.fillStyle = '#172e22'; context.font = 'bold 18px sans-serif'; context.fillText(name,x+12,y-11);}
}
async function saveSetup() {
  if (!state.calibration || Object.keys(state.anchors).length !== 4 || state.busy) return;
  setBusy(true); $('save-setup').disabled = true; $('calibration-state').textContent = 'VERIFYING HOLES';
  try {
    const result = await api('/setups', {method:'POST', body:{name:$('setup-name').value.trim(), image_base64:state.calibration.image, columns:state.calibration.columns, anchors:state.anchors}});
    state.setupId = result.id; savePreference('sherlock.live.setup', result.id);
    const bitmap = new Image(); bitmap.src = `data:image/jpeg;base64,${result.image_base64}`; await bitmap.decode(); state.calibration.bitmap = bitmap; state.anchors = {}; drawCalibration();
    $('calibration-state').textContent = 'CALIBRATION SAVED'; $('calibration-instructions').textContent = 'Review the grid and labels. If these match your physical board, you’re ready to inspect.';
    await refreshStatus(); notice('Board calibration saved. Choose a reference circuit, then start a live check.', true);
  } catch (error) {$('calibration-state').textContent = 'CHECK YOUR ANCHORS'; notice(errorText(error)); renderAnchors();} finally {setBusy(false);}
}
function addPart({type='wire',from='a1',to='a5'} = {}) {
  const row = document.createElement('div'); row.className = 'part-editor-row';
  row.innerHTML = `<select aria-label="Component type"><option value="wire">Wire</option><option value="resistor">Resistor</option></select><input aria-label="From hole" placeholder="a1" value="${esc(from)}" maxlength="3" required><input aria-label="To hole" placeholder="a5" value="${esc(to)}" maxlength="3" required><button type="button" aria-label="Remove component">×</button>`;
  row.querySelector('select').value = type; row.querySelector('button').addEventListener('click', () => row.remove()); $('reference-parts').append(row);
}
async function saveReference(event) {
  event.preventDefault(); if (state.busy) return;
  try {
    const rows = [...$('reference-parts').children].map(row => ({type:row.querySelector('select').value, from:row.querySelectorAll('input')[0].value, to:row.querySelectorAll('input')[1].value}));
    const body = makeReference($('reference-name').value, rows, {VCC:$('terminal-vcc').value, GND:$('terminal-gnd').value});
    setBusy(true); $('save-reference').disabled = true;
    const result = await api('/references', {method:'POST',body}); state.referenceId = result.id; savePreference('sherlock.live.reference', result.id);
    await refreshStatus(); $('reference-save-message').textContent = `Saved “${result.name}”. Ready to use in Live wiring.`; notice('Reference circuit saved.',true);
  } catch (error) {notice(errorText(error));} finally {setBusy(false); $('save-reference').disabled = false;}
}
async function loadHistory() {
  $('refresh-history').disabled = true;
  try {
    const {inspections} = await api('/history');
    $('history-list').innerHTML = inspections.length ? inspections.map(item => {
      const date = new Date(item.created_at), status = item.status || 'UNCERTAIN';
      const tone = status === 'CORRECT' ? 'green' : status === 'INCORRECT' ? 'red' : '';
      return `<div class="history-row"><div><strong>${esc(item.mode === 'preview' ? 'Endpoint preview' : item.mode === 'still' ? 'Photo inspection' : 'Live inspection')}</strong><small>${esc(item.message)}</small></div><span class="result-badge ${tone}">${esc(status === 'CORRECT' ? 'MATCHED AT CAPTURE' : status === 'INCORRECT' ? 'MISMATCH AT CAPTURE' : status)}</span><span>${(item.parts?.length || item.detections?.length || 0).toString().padStart(2,'0')}</span><time datetime="${esc(item.created_at)}">${esc(date.toLocaleDateString(undefined,{month:'short',day:'numeric'}))}<br>${esc(date.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}))}</time></div>`;
    }).join('') : '<div class="quiet-empty">No inspections yet. Your first connection is a good place to start.</div>';
  } catch (error) {notice(errorText(error));} finally {$('refresh-history').disabled = false;}
}
function renderIllustration() {
  const holes = []; for (let row=0;row<10;row++) for (let col=0;col<30;col++) holes.push(`<rect x="${32+col*11}" y="${row<5?40+row*9:104+(row-5)*9}" width="2.7" height="3.2" rx=".5" fill="#7e8778" opacity=".7"/>`);
  const numbers = [1,5,10,15,20,25,30].map(n=>`<text x="${31+(n-1)*11}" y="30" fill="#969d8a" font-size="5" font-family="monospace">${n}</text>`).join('');
  $('board-illustration').innerHTML = `<svg viewBox="0 0 390 172" xmlns="http://www.w3.org/2000/svg"><rect x="5" y="8" width="380" height="160" rx="9" fill="#0b1f15" opacity=".4"/><rect x="4" y="2" width="380" height="160" rx="7" fill="#dce2cd"/><rect x="10" y="8" width="368" height="147" rx="5" fill="#edf0de"/><path d="M19 20h350" stroke="#c1776c" stroke-width=".65"/><path d="M19 149h350" stroke="#8baba3" stroke-width=".65"/><rect x="21" y="85" width="348" height="8" rx="3" fill="#ccd3bb"/><path d="M25 86h339" stroke="#b9c3a6" stroke-width="1"/>${holes.join('')}${numbers}<g font-size="5" fill="#a7ae9a" font-family="monospace"><text x="21" y="44">a</text><text x="21" y="80">e</text><text x="21" y="108">f</text><text x="21" y="144">j</text></g><path d="M78 61C78 15 123 5 164 14S232 25 232 61" fill="none" stroke="#6d8a4e" stroke-width="7"/><path d="M78 61C78 15 123 5 164 14S232 25 232 61" fill="none" stroke="#d4ec91" stroke-width="4.5"/><path d="M276 123C276 98 297 97 308 110S335 143 345 122" fill="none" stroke="#d4b87d" stroke-width="4"/><circle cx="78" cy="61" r="4.5" fill="#d7f698" stroke="#5c7841" stroke-width="1.5"/><circle cx="232" cy="61" r="4.5" fill="#d7f698" stroke="#5c7841" stroke-width="1.5"/></svg>`;
}

$('camera-button').addEventListener('click',connectCamera); $('empty-camera').addEventListener('click',connectCamera);
$('camera-tab').addEventListener('click',()=>{setSource('camera'); if (state.stream) {state.result=null; showRawCamera(); renderResult();} else connectCamera();});
for (const id of ['upload-button','empty-upload','photo-tab']) $(id).addEventListener('click',()=>{if (!state.busy && !state.running) $('photo-input').click();});
$('photo-input').addEventListener('change',event=>inspectPhoto(event.target.files[0])); $('live-button').addEventListener('click',startLive);
$('take-photo-button').addEventListener('click',takePhoto);
for (const [id,key] of [['reference-select','referenceId'],['setup-select','setupId']]) $(id).addEventListener('change',()=>{stopLive(); state[key]=$(id).value; savePreference(`sherlock.live.${key==='setupId'?'setup':'reference'}`,state[key]); state.result=null; if(state.stream)showRawCamera(); renderResult(); updateControls();});
$('calibration-upload').addEventListener('click',()=>{if (!state.busy) $('calibration-input').click();}); $('calibration-input').addEventListener('change',event=>prepareCalibration(event.target.files[0]));
$('setup-columns').addEventListener('change',()=>{state.calibration=null;state.anchors={};$('calibration-canvas').hidden=true;$('calibration-placeholder').hidden=false;$('calibration-state').textContent='WAITING FOR PHOTO';$('calibration-instructions').textContent='Choose the empty-board photo for this board size.';renderAnchors();});
$('calibration-canvas').addEventListener('click',event=>{if(!state.calibration||state.busy)return;const names=anchorNames(),count=Object.keys(state.anchors).length;if(count>=4)return;const canvas=$('calibration-canvas'),rect=canvas.getBoundingClientRect();state.anchors[names[count]]=[(event.clientX-rect.left)*canvas.width/rect.width,(event.clientY-rect.top)*canvas.height/rect.height];drawCalibration();renderAnchors();});
$('reset-anchors').addEventListener('click',()=>{state.anchors={};drawCalibration();renderAnchors();}); $('save-setup').addEventListener('click',saveSetup);
$('add-part').addEventListener('click',()=>addPart({from:'',to:''})); $('reference-form').addEventListener('submit',saveReference);
$('reference-library').addEventListener('click',event=>{const button=event.target.closest('[data-reference]');if(!button)return;state.referenceId=button.dataset.reference;savePreference('sherlock.live.reference',state.referenceId);$('reference-select').value=state.referenceId;location.hash='inspect';updateControls();});
$('refresh-history').addEventListener('click',loadHistory);
$('connection-settings').addEventListener('click',()=>{$('api-address').value=state.backend;$('connection-dialog').showModal();});
$('save-connection').addEventListener('click',async()=>{if(state.busy)return;const address=$('api-address').value.trim().replace(/\/$/,'');try{if(address){const url=new URL(address);if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)throw new Error('Use an HTTP or HTTPS backend address without credentials, query, or fragment.');}stopLive();disconnectCamera();setBusy(true);state.backend=address;state.setupId='';state.referenceId='';state.result=null;state.calibration=null;state.anchors={};$('calibration-canvas').hidden=true;$('calibration-placeholder').hidden=false;renderAnchors();savePreference('sherlock.live.backend',address);$('connection-dialog').close();$('notice').hidden=true;renderResult();await refreshStatus();}catch(error){notice(errorText(error));}finally{setBusy(false);}});
window.addEventListener('hashchange',navigate); document.addEventListener('visibilitychange',()=>{if(document.hidden)stopLive();});
window.addEventListener('pagehide',()=>{disconnectCamera();});
setInterval(()=>{if(state.running&&state.result)renderResult();},250);
renderIllustration(); addPart(); renderAnchors(); navigate(); refreshStatus();
