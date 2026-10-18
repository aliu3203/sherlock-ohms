import test from 'node:test';
import assert from 'node:assert/strict';
import {presentVerdict, makeReference, escapeHTML} from '../live-state.mjs';

const correct = {status:'CORRECT', passed:true, mode:'live', stable_frames:3, required_stable_frames:3, valid_for_ms:2000, message:'Matches.'};

test('green needs a live stream, stable frames, and a fresh capture', () => {
  assert.equal(presentVerdict(correct, {live:true, capturedAt:100, now:1500}).tone,'green');
  for (const options of [{live:false,capturedAt:100,now:1500},{live:true,capturedAt:100,now:2101},{live:true,capturedAt:100,now:50}]) {
    assert.notEqual(presentVerdict(correct,options).tone,'green');
  }
  assert.notEqual(presentVerdict({...correct,stable_frames:2},{live:true,capturedAt:100,now:1000}).tone,'green');
  assert.notEqual(presentVerdict({...correct,passed:false},{live:true,capturedAt:100,now:1000}).tone,'green');
});
test('upload plus inference time counts toward expiration; server cannot extend freshness', () => {
  assert.equal(presentVerdict({...correct,valid_for_ms:9000},{live:true,capturedAt:0,now:2001}).expired,true);
  assert.equal(presentVerdict({...correct,valid_for_ms:500},{live:true,capturedAt:100,now:601}).expired,true);
});
test('still images remain amber even if their candidate matches', () => {
  const result = presentVerdict({...correct,mode:'still',candidate_status:'CORRECT'});
  assert.equal(result.tone,'amber'); assert.match(result.message,/photo/); assert.match(result.label,/confirmation/);
  assert.equal(presentVerdict({...correct,mode:'still'}, {live:true,capturedAt:100,now:500}).tone,'amber');
});
test('expired incorrect verdict and unknown status cannot masquerade as current certainty', () => {
  assert.equal(presentVerdict({...correct,status:'INCORRECT',passed:false},{live:true,capturedAt:10,now:2011}).tone,'amber');
  assert.equal(presentVerdict({...correct,status:'UNRECOGNIZED'},{live:true,capturedAt:10,now:200}).tone,'amber');
});
test('equivalent strip holes remain valid reference inputs but duplicate physical holes fail', () => {
  const reference = makeReference('  Two wires  ',[{type:'wire',from:' A1 ',to:'a5'},{type:'wire',from:'b5',to:'a8'}],{VCC:'e1',GND:''});
  assert.equal(reference.name,'Two wires'); assert.deepEqual(reference.terminals,{VCC:'e1'});
  assert.equal(reference.parts[1].pins['0'],'b5');
  assert.throws(()=>makeReference('Bad',[{type:'wire',from:'a1',to:'a1'}]),/same physical hole/);
});
test('invalid holes, unsupported parts and invalid terminals are rejected before upload', () => {
  for (const hole of ['a0','a64','k1','<img>','1a']) assert.throws(()=>makeReference('Bad',[{type:'wire',from:hole,to:'a5'}]));
  assert.throws(()=>makeReference('LED',[{type:'led',from:'a1',to:'a5'}]),/wires and resistors/);
  assert.throws(()=>makeReference('Power',[{type:'wire',from:'a1',to:'a5'}],{VCC:'rail1'}),/VCC/);
});
test('saved names and API errors cannot inject markup through escaped UI templates', () => {
  assert.equal(escapeHTML('<img src=x onerror="alert(1)">'), '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
});
