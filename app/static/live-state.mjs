export const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

// A live verdict expires from the start of capture, including upload and inference.
// A stopped stream, a still image, and an unknown server status cannot turn green.
export function presentVerdict(result, {live = false, now = 0, capturedAt = 0} = {}) {
  if (!result) return {tone:'neutral', title:'Ready when you are.', symbol:'⌁', message:'Connect a camera or add a photo to see what’s on your board.', label:'No circuit verdict yet'};
  const status = result.status;
  const fresh = live && Number.isFinite(capturedAt) && now >= capturedAt && now - capturedAt <= Math.min(2000, Math.max(0, Number(result.valid_for_ms) || 2000));
  if (live && !fresh) return {tone:'amber', title:'Waiting for a fresh frame.', symbol:'⌁', message:'The last live result has expired. Keep the camera running for current feedback.', label:'Expired · no current verdict', expired:true};
  if (fresh && result.mode !== 'still' && status === 'CORRECT' && result.passed === true && result.stable_frames >= Math.max(2, result.required_stable_frames || 3)) return {tone:'green', title:'A good connection.', symbol:'✓', message:result.message, label:'Visible connectivity matches'};
  if (fresh && result.mode !== 'still' && status === 'INCORRECT' && result.stable_frames >= Math.max(2, result.required_stable_frames || 3)) return {tone:'red', title:'Something doesn’t line up.', symbol:'↯', message:result.message, label:'Stable wiring mismatch'};
  if (status === 'UNCALIBRATED') return {tone:'amber', title:'Found the endpoints.', symbol:'⌁', message:result.message, label:'Preview only · board setup needed'};
  if (result.mode === 'still') return {tone:'amber', title:'One frame, a closer look.', symbol:'⌁', message: result.candidate_status === 'CORRECT' ? 'This photo’s visible connections match the reference. Start live capture to check consistency across frames.' : result.candidate_status === 'INCORRECT' ? 'This photo suggests a wiring mismatch. Start live capture to confirm it across frames.' : result.message, label:'Photo check · awaiting live confirmation'};
  return {tone:'amber', title:status === 'INCOMPLETE' ? 'A few pieces are missing.' : 'Taking a closer look.', symbol:'⌁', message:result.message || 'A reliable current verdict is not available.', label:status === 'INCOMPLETE' ? 'Incomplete · no pass' : 'Uncertain · no current verdict'};
}

export function makeReference(name, rows, terminals = {}) {
  const hole = /^[a-j](?:[1-9]|[1-5][0-9]|6[0-3])$/;
  if (!name.trim() || !rows.length) throw new Error('Give your circuit a name and add at least one component.');
  const occupied = new Set();
  const counts = {wire:0, resistor:0};
  const parts = rows.map(({type, from, to}) => {
    if (!(type in counts)) throw new Error('This model supports wires and resistors.');
    const pins = [from, to].map(value => value.trim().toLowerCase());
    for (const pin of pins) {
      if (!hole.test(pin)) throw new Error('Use hole names from a1 to j63, following your printed board labels.');
      if (occupied.has(pin)) throw new Error(`Two pins cannot occupy the same physical hole (${pin}).`);
      occupied.add(pin);
    }
    return {id:`${type === 'wire' ? 'W' : 'R'}${++counts[type]}`, type, pins:{'0':pins[0], '1':pins[1]}};
  });
  const cleaned = {};
  for (const [key, value] of Object.entries(terminals)) {
    const pin = value.trim().toLowerCase();
    if (pin && !hole.test(pin)) throw new Error(`${key} needs a valid hole name, such as a1.`);
    if (pin) cleaned[key] = pin;
  }
  return {name:name.trim(), parts, terminals:cleaned};
}
