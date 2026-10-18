// Deterministic, explicitly simulated fixtures. No electrical verification runs here.
export const SEED_REFERENCE = {
  id: 'demo-led', name: 'LED + current-limiting resistor',
  rails: {VCC: 'top_plus', GND: 'top_minus'},
  parts: [
    {id: 'W_VCC', type: 'wire', pins: {'0': 'top_plus_5', '1': 'a5'}},
    {id: 'R1', type: 'resistor', pins: {'0': 'a5', '1': 'a10'}},
    {id: 'LED1', type: 'led', pins: {anode: 'd10', cathode: 'd15'}},
    {id: 'W_GND', type: 'wire', pins: {'0': 'a15', '1': 'top_minus_15'}},
  ],
};
export const SCENARIOS = ['exact', 'short', 'reversed', 'offbyone', 'equivalent'];
export function createLUT() {
  const holes = {}, pitch = 20.32;
  for (let col = 1; col <= 63; col++) {
    const x = col * pitch;
    for (const [row, y] of [['top_plus', pitch], ['top_minus', 2*pitch], ['bottom_plus', 18*pitch], ['bottom_minus', 19*pitch]]) holes[`${row}_${col}`] = {x, y};
    for (const [i, row] of [...'abcdefghij'].entries()) holes[`${row}${col}`] = {x, y: (i < 5 ? i+4 : i+6)*pitch};
  }
  return holes;
}
export function normalizeLUT(value) {
  const lut = value?.holes || value || {}, result = {};
  for (const [name, point] of Object.entries(lut)) {
    const x = Number(Array.isArray(point) ? point[0] : point?.x);
    const y = Number(Array.isArray(point) ? point[1] : point?.y);
    if (Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= 1400 && y >= 0 && y <= 520) result[name] = {x, y};
  }
  return result;
}
export function validateReference(value) {
  if (!value || typeof value.name !== 'string' || !value.name.trim() || value.name.length > 120) throw new Error('Give the reference a name (up to 120 characters).');
  if (!value.rails || typeof value.rails !== 'object' || Array.isArray(value.rails) || Object.values(value.rails).some(v => typeof v !== 'string')) throw new Error('rails must map electrical names to rail names.');
  if (!Array.isArray(value.parts) || value.parts.length === 0 || value.parts.length > 200) throw new Error('Include 1–200 parts in the circuit.');
  const ids = new Set(), types = ['wire', 'resistor', 'led', 'cap_ceramic', 'cap_electrolytic', 'button', 'ic_dip'];
  for (const p of value.parts) {
    if (!p || typeof p.id !== 'string' || !p.id || ids.has(p.id)) throw new Error('Every part needs a unique id.');
    if (!types.includes(p.type)) throw new Error(`Unsupported part type: ${p.type}`);
    if (!p.pins || typeof p.pins !== 'object' || Array.isArray(p.pins) || Object.keys(p.pins).length !== 2 || Object.values(p.pins).some(v => typeof v !== 'string' || !v)) throw new Error(`Part ${p.id} needs two named pins.`);
    const names = ['led', 'cap_electrolytic'].includes(p.type) ? ['anode', 'cathode'] : ['0', '1'];
    if (!names.every(n => Object.hasOwn(p.pins, n))) throw new Error(`Part ${p.id} requires pins ${names.join(' and ')}.`);
    ids.add(p.id);
  }
  return value;
}
export function partStatus(part, findings) {
  const related = findings.filter(f => f.part_id === part.id || (!f.part_id && f.severity === 'critical'));
  if (related.some(f => f.severity === 'critical')) return 'critical';
  if (part.ambiguous_pins?.length || related.some(f => f.code === 'UNCERTAIN')) return 'uncertain';
  if (related.some(f => f.severity === 'warning')) return 'warning';
  return 'valid';
}
export function makeDemoResult(scenario = 'exact', mode = 'connectivity') {
  if (!SCENARIOS.includes(scenario)) throw new Error('Unknown demo scenario.');
  if (!['connectivity', 'placement'].includes(mode)) throw new Error('Unknown verification mode.');
  const lut = createLUT(), parts = structuredClone(SEED_REFERENCE.parts), findings = [];
  const fault = (code, message, part_id, holes = [], expected_holes = []) => findings.push({code, severity: 'critical', message, part_id, holes, expected_holes});
  if (scenario === 'short') {
    parts.push({id: 'W_SHORT', type: 'wire', pins: {'0': 'top_plus_35', '1': 'top_minus_35'}});
    fault('POWER_SHORT', 'A jumper directly connects the positive supply to ground. Remove this wire before applying power.', 'W_SHORT', ['top_plus_35', 'top_minus_35']);
  } else if (scenario === 'reversed') {
    parts[2].pins = {anode: 'd15', cathode: 'd10'};
    fault('REVERSED_POLARITY', 'The LED is facing the wrong way. Swap its anode and cathode so current flows toward ground.', 'LED1', ['d15', 'd10'], ['d10', 'd15']);
  } else if (scenario === 'offbyone') {
    parts[2].pins.cathode = 'd16';
    fault(mode === 'placement' ? 'MISPLACED_PIN' : 'OPEN_CONNECTION', 'The LED cathode is in d16 instead of d15. Move it one hole left to reconnect ground.', 'LED1', ['d16'], ['d15']);
  } else if (scenario === 'equivalent') {
    parts[0].pins = {'0': 'top_plus_20', '1': 'a20'};
    parts[1].pins = {'0': 'a20', '1': 'a25'};
    parts[2].pins = {anode: 'i25', cathode: 'i30'};
    parts[3].pins = {'0': 'j30', '1': 'top_minus_30'};
    parts.push({id: 'W_BRIDGE', type: 'wire', pins: {'0': 'e25', '1': 'f25'}});
    if (mode === 'placement') {
      for (const index of [1, 2]) fault('MISPLACED_PIN', `${parts[index].id} is in different holes from the golden layout. This circuit is electrically equivalent; choose connectivity mode to accept it.`, parts[index].id, Object.values(parts[index].pins), Object.values(SEED_REFERENCE.parts[index].pins));
    }
  }
  const detections = parts.map(p => {
    const points = Object.values(p.pins).map(h => lut[h]);
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    return {id: p.id, type: p.type, conf: .98, bbox: [Math.min(...xs)-10, Math.min(...ys)-10, Math.max(...xs)+10, Math.max(...ys)+10], keypoints: points.map(p => ({...p, conf: .98, vis: 2}))};
  });
  const snapped_parts = parts.map((p, i) => ({...p, bbox: detections[i].bbox, conf: .98, ambiguous_pins: []}));
  const passed = !findings.some(f => f.severity === 'critical');
  return {id: `demo-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, timestamp: new Date().toISOString(), reference_id: SEED_REFERENCE.id, reference_name: SEED_REFERENCE.name, simulation: true, scenario, mode, lut, detections, verdict: {passed, light_color: passed ? 'GREEN' : 'RED', snapped_parts, findings, actual_netlist: {}}, elapsed_ms: 0};
}
