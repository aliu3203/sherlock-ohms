import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {SEED_REFERENCE, SCENARIOS, createLUT, normalizeLUT, makeDemoResult, validateReference, partStatus} from '../demo.mjs';
import {normalizeReference, normalizeHistory, normalizeCalibration, normalizeInspection, eventProgress} from '../api.mjs';

test('all seeded scenarios produce expected connectivity verdicts and valid geometry', () => {
  for (const scenario of SCENARIOS) {
    const result = makeDemoResult(scenario);
    assert.equal(result.simulation, true);
    assert.equal(result.verdict.passed, ['exact','equivalent'].includes(scenario));
    assert.equal(result.verdict.light_color, result.verdict.passed ? 'GREEN' : 'RED');
    for (const part of result.verdict.snapped_parts) {
      assert.ok(part.bbox.every(Number.isFinite));
      assert.equal(Object.values(part.pins).length, 2);
      for (const hole of Object.values(part.pins)) assert.ok(result.lut[hole]);
    }
  }
});
test('equivalent layout is accepted only in connectivity mode', () => {
  assert.equal(makeDemoResult('equivalent','connectivity').verdict.passed, true);
  const placed = makeDemoResult('equivalent','placement');
  assert.equal(placed.verdict.passed, false);
  assert.ok(placed.verdict.findings.every(f=>f.code==='MISPLACED_PIN'));
  const bridge=placed.verdict.snapped_parts.find(p=>p.id==='W_BRIDGE');
  assert.deepEqual(Object.values(bridge.pins), ['e25','f25']);
});
test('off-by-one and reversed scenarios have real expected hole locations for ghost vectors', () => {
  const off=makeDemoResult('offbyone');
  assert.deepEqual(off.verdict.findings[0].holes,['d16']);
  assert.deepEqual(off.verdict.findings[0].expected_holes,['d15']);
  assert.ok(off.lut.d16.x>off.lut.d15.x);
  assert.equal(makeDemoResult('reversed').verdict.findings[0].code,'REVERSED_POLARITY');
  assert.equal(makeDemoResult('short').verdict.findings[0].code,'POWER_SHORT');
});
test('demo results do not mutate the golden reference', () => {
  const before=JSON.stringify(SEED_REFERENCE);
  SCENARIOS.forEach(s=>makeDemoResult(s));
  assert.equal(JSON.stringify(SEED_REFERENCE),before);
});
test('LUT accepts calibrated object and tuple formats, rejects invalid coordinates', () => {
  assert.equal(Object.keys(createLUT()).length, 882);
  assert.deepEqual(normalizeLUT({holes:{a1:[20,80],a2:{x:40,y:80},bad:{x:NaN,y:5},outside:[1401,1]}}),{a1:{x:20,y:80},a2:{x:40,y:80}});
});
test('critical, warning, uncertain, and matched outline states are distinct', () => {
  const p={id:'R1'};
  assert.equal(partStatus(p,[]),'valid');
  assert.equal(partStatus(p,[{part_id:'R1',severity:'warning',code:'EXTRA_PART'}]),'warning');
  assert.equal(partStatus(p,[{part_id:'R1',severity:'warning',code:'UNCERTAIN'}]),'uncertain');
  assert.equal(partStatus({...p,ambiguous_pins:['0']},[]),'uncertain');
  assert.equal(partStatus(p,[{part_id:null,severity:'critical',code:'POWER_SHORT'}]),'critical');
});
test('reference validation catches duplicate IDs and invalid LED pin roles', () => {
  assert.equal(validateReference(structuredClone(SEED_REFERENCE)).name,SEED_REFERENCE.name);
  const duplicate=structuredClone(SEED_REFERENCE);duplicate.parts[1].id=duplicate.parts[0].id;
  assert.throws(()=>validateReference(duplicate),/unique/);
  const invalid=structuredClone(SEED_REFERENCE);invalid.parts[2].pins={'0':'a1','1':'a2'};
  assert.throws(()=>validateReference(invalid),/anode/);
});
test('adapts real backend nested reference and flat inspection schemas', () => {
  const ref=normalizeReference({id:'led_blink',name:'LED Blink',data:{name:'LED Blink',rails:{},parts:[]}});
  assert.deepEqual(ref.parts,[]);
  const record={inspection_id:'abc',reference_id:'led_blink',passed:false,status:'done',snapped_parts:[],findings:[{code:'MISSING_PART'}],timestamp:'2026-10-17T20:00:00Z'};
  const result=normalizeInspection(record);
  assert.equal(result.id,'abc');assert.equal(result.verdict.passed,false);
  assert.equal(result.rectified_image_url,'/api/inspections/abc/image?layer=rectified');
  assert.equal(normalizeHistory({inspections:[record]},[ref])[0].reference_name,'LED Blink');
});
test('backend quality abort and errors never become passing circuit verdicts', () => {
  const aborted=normalizeInspection({inspection_id:'x',status:'aborted',passed:false,quality:{passed:false,failures:['brightness']},snapped_parts:[]});
  assert.equal(aborted.aborted,true);assert.equal(aborted.verdict.passed,false);
  const error=normalizeInspection({inspection_id:'x',status:'error',passed:false,error:'Camera failed'});
  assert.equal(error.aborted,true);assert.equal(error.verdict,null);
  assert.throws(()=>normalizeInspection({status:'running',passed:null}),/completed verdict/);
});
test('registration does not falsely imply hole LUT calibration', () => {
  const calibration=normalizeCalibration({data:{has_homography:true,quality:{passed:true}},updated_at:'2026-10-17'});
  assert.equal(calibration.registered,true);assert.equal(calibration.calibrated,false);assert.equal(calibration.lut,undefined);
});
test('all actual SSE progress stages map to increasing progress', () => {
  const stages=['capturing','registering','detecting','checking'].map(stage=>eventProgress({stage}));
  assert.deepEqual(stages.map(s=>s.percent),[8,25,45,80]);
  assert.equal(eventProgress({stage:'detecting',progress:.55}).percent,55);
});
test('frontend selectors match HTML and assets exist', () => {
  const html=fs.readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const js=fs.readFileSync(new URL('../app.js',import.meta.url),'utf8');
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
  assert.equal(ids.length,new Set(ids).size,'HTML IDs must be unique');
  for(const match of js.matchAll(/\$\('([^']+)'\)/g))assert.ok(ids.includes(match[1]),`Missing element #${match[1]}`);
  for(const asset of ['styles.css','app.js','demo.mjs','api.mjs','board-base.svg'])assert.ok(fs.existsSync(new URL(`../${asset}`,import.meta.url)));
  assert.match(html,/viewBox="0 0 1400 520"/);
});
