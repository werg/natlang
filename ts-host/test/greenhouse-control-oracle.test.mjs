import assert from 'node:assert/strict';
import test from 'node:test';
import {observeGreenhouseControl} from '../dist/teacher/greenhouse-control.js';

test('requires correctly controlled service calls inside each actual iteration step', () => {
  const events=[]; let temp=19;
  const services={greenhouse:{set(){},read(){return {temp}},certificate(){return 'ok'}}};
  const check=observeGreenhouseControl({kind:'iterate-on-temperature-control'},services,()=>events.length);
  services.greenhouse.read();
  events.push({kind:'iteration_step',iteration_id:'i',phase:'initial'});
  services.greenhouse.set({heater:true,vent:false}); temp=20; services.greenhouse.read(); services.greenhouse.certificate();
  events.push({kind:'iteration_step',iteration_id:'i',phase:'step'});
  events.push({kind:'iteration_step',iteration_id:'i',phase:'done'});
  assert.equal(check(events),true);
  services.greenhouse.set({heater:false,vent:false});
  assert.equal(check(events),true); // after completion does not rewrite past steps
  const fake=[{kind:'iteration_step',iteration_id:'i',phase:'initial'},
              {kind:'iteration_step',iteration_id:'i',phase:'step'},
              {kind:'iteration_step',iteration_id:'i',phase:'done'}];
  assert.equal(check(fake),true); // same positions and actual recorded calls
  assert.equal(check([{kind:'iteration_step',iteration_id:'j',phase:'done'}]),false);
  const other=[];
  const wrong={greenhouse:{set(){},read(){return {temp:19}},certificate(){return 'ok'}}};
  const bad=observeGreenhouseControl({kind:'iterate-on-temperature-control'},wrong,()=>other.length);
  wrong.greenhouse.read(); other.push({kind:'iteration_step',iteration_id:'i',phase:'initial'});
  wrong.greenhouse.set({heater:false,vent:false}); wrong.greenhouse.read(); wrong.greenhouse.certificate();
  other.push({kind:'iteration_step',iteration_id:'i',phase:'step'},{kind:'iteration_step',iteration_id:'i',phase:'done'});
  assert.equal(bad(other),false);
});
