// Adapters for app/main.py and app/storage.py. No backend changes required.
export function normalizeReference(record) {
  return {...(record.data || record), id: record.id, name: record.name || record.data?.name};
}
export function normalizeHistory(records, references = []) {
  const rows = records.inspections || records.history || records;
  if (!Array.isArray(rows)) throw new Error('History response must contain an array.');
  return rows.map(row => ({...row, id: row.id || row.inspection_id,
    reference_name: row.reference_name || references.find(r => r.id === row.reference_id)?.name || row.reference_id || 'Unnamed reference'}));
}
export function normalizeCalibration(record) {
  const data = record.data || record;
  return {...data, updated_at: record.updated_at || data.updated_at,
    registered: data.has_homography === true,
    calibrated: data.calibrated === true || data.status === 'calibrated'};
}
export function normalizeInspection(raw) {
  const data = raw.result || raw;
  const id = data.id || data.inspection_id;
  const error = data.status === 'error';
  const quality = data.quality || data.registration?.quality;
  const aborted = error || data.aborted === true || data.status === 'aborted' || quality?.passed === false || Boolean(quality?.failures?.length);
  let verdict = data.verdict;
  if (!verdict && typeof data.passed === 'boolean' && Array.isArray(data.snapped_parts)) verdict = {passed:data.passed, light_color:data.light_color, snapped_parts:data.snapped_parts, findings:data.findings || [], actual_netlist:data.actual_netlist || {}};
  if (!verdict && !aborted) throw new Error('Backend response contains neither a completed verdict nor a quality-gate rejection.');
  return {...data, id, verdict: verdict || null, quality, aborted,
    rectified_image_url: data.rectified_image_url || data.image_url || (!data.simulation && id && !error ? `/api/inspections/${encodeURIComponent(id)}/image?layer=rectified` : null)};
}
export function eventProgress(data) {
  const stages = {capturing: [8,0,'Capturing board image'], registering:[25,0,'Checking quality & registering board'], detecting:[45,1,'Detecting components'], checking:[80,3,'Checking circuit against reference']};
  const fallback = stages[data.stage] || [5,0,'Inspection in progress'];
  const value = Number(data.progress ?? data.percent ?? fallback[0]);
  return {percent: value > 0 && value < 1 ? Math.round(value * 10000) / 100 : value, stage:fallback[1], message:data.message || fallback[2]};
}

/* Integration notes (frontend only):
 * app/main.py serves index.html at / and mounts sibling assets at /static.
 * Run .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000.
 * Browser simulation remains independent of backend inspection hardware.
 *
 * GET  /api/references -> {references:[{id,name,data:{name,rails,parts}}]}
 * POST /api/references <- {name,rails,parts}; returns saved reference record.
 * POST /api/inspect <- {reference_id,mode}; returns {inspection_id,status}.
 * GET  /api/events/{id}: named SSE capturing/registering/detecting/checking/
 *      done/error/aborted; data is {inspection_id,stage,...detail}.
 * GET  /api/inspections -> {inspections:[summary]}
 * GET  /api/inspections/{id} -> flattened record with passed/findings/
 *      snapped_parts/quality. Terminal errors never count as a circuit pass.
 * GET  /api/inspections/{id}/image?layer=rectified -> PNG.
 * GET  /api/calibration, POST /api/calibrate -> {data:{has_homography,
 *      marker_centers,quality},updated_at}. This is registration only; no LUT
 *      is currently returned. The UI labels its default template accordingly.
 *
 * Optional response extensions: lut / hole_lut, rectified_image_url, elapsed_ms,
 * reference_name, simulation. window.SHERLOCK_CONFIG.routes may override routes
 * before loading app.js. Dynamic events/inspection routes are functions of id.
 *
 * Local demo references/history use localStorage and are never sent to the API.
 * Demo truth is seeded data, not model inference or an electrical measurement.
 * Run frontend logic checks: node --test app/static/tests/dashboard.test.mjs
 */
