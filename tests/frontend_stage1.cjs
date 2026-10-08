// Execute the actual inline script with a minimal DOM, without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    style: {}, disabled: false, value: '', files: [], src: 'stale', innerHTML: 'stale',
    classList: {add() {}, remove() {}}, listeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    removeAttribute(name) { delete this[name]; },
    replaceChildren() { this.innerHTML = ''; },
  });
  return elements.get(id);
}
const revoked = [], alerts = [];
let counter = 0;
const context = vm.createContext({
  document: {getElementById: element},
  window: {addEventListener() {}},
  URL: {createObjectURL() {return `blob:${++counter}`;}, revokeObjectURL(url) {revoked.push(url);}},
  FormData: class {append() {}},
  alert(message) {alerts.push(message);},
});
vm.runInContext(script, context);
const realRenderOutput = context.renderOutput;
let rendered = 0;
context.renderOutput = () => {rendered++; element('resultsWrap').style.display = 'block';};
const run = code => vm.runInContext(code, context);
(async () => {
  element('fileInput').files = [{}];
  run('showSelectedFile()');
  run('showSelectedFile()');
  assert.deepEqual(revoked, ['blob:1']);
  let finish;
  context.fetch = () => new Promise(resolve => {finish = resolve;});
  const pending = run('executeScan()');
  assert.equal(element('resetBtn').disabled, true);
  assert.equal(element('fileInput').disabled, true);
  assert.equal(element('resultsWrap').style.display, 'none');
  assert.equal(element('pipeSource').src, undefined);
  run('clearScan()');
  assert.equal(element('imagePreview').src, 'blob:2');
  finish({status: 502, ok: false, headers: {get() {return 'text/html';}}});
  await pending;
  assert.match(alerts.pop(), /unreadable response \(HTTP 502\)/);
  assert.equal(element('resetBtn').disabled, false);
  assert.equal(element('resultsWrap').style.display, 'none');
  context.fetch = async () => ({status: 422, ok: false, headers: {get() {return 'application/json';}}, json: async () => ({error: 'Retake photo'})});
  await run('executeScan()');
  assert.match(alerts.pop(), /Retake photo/);
  context.fetch = async () => ({status: 200, ok: true, headers: {get() {return 'application/json';}}, json: async () => {throw new Error('syntax');}});
  await run('executeScan()');
  assert.match(alerts.pop(), /invalid response data/);
  context.fetch = async () => ({status: 200, ok: true, headers: {get() {return 'application/json';}}, json: async () => ({prediction: 'Healthy', probabilities: {}, advisory: {}, tissue_stats: {}})});
  await run('executeScan()');
  assert.equal(rendered, 1);
  assert.equal(element('resultsWrap').style.display, 'block');
  run('clearScan()');
  assert.deepEqual(revoked, ['blob:1', 'blob:2']);
  assert.equal(element('imagePreview').src, undefined);
  assert.equal(element('resultsWrap').style.display, 'none');
  const result = {
    prediction: 'Healthy', visual_stress_indicator: 26, best_model_name: 'Test model',
    probabilities: {Healthy: .8, Disease_Confounded: .2},
    multi_model_predictions: {Other: 'Disease_Confounded'},
    tissue_stats: {healthy_pct: 80, chlorosis_pct: 10, necrosis_pct: 10},
    advisory: {guidance: 'Check soil moisture directly.', action_items: []},
    features: {solidity: .8},
    feature_comparisons: [{label: 'Solidity', value: .8, dataset_mean: .7, z_score: 1,
      interpretation: 'Compared with the pooled dataset across all classes.'}],
  };
  realRenderOutput(result);
  assert.equal(element('indicatorValue').textContent, '26');
  assert.equal(element('indicatorWrap').hidden, false);
  assert.equal(element('indicatorUnavailable').hidden, true);
  assert.match(element('measurementsContainer').innerHTML, /Dataset average: 0.7/);
  assert.match(element('probabilityBars').innerHTML, /Disease-like or other confounding symptoms suspected/);
  assert.match(element('consensusEntries').innerHTML, /Disease-like or other confounding symptoms suspected/);
  realRenderOutput({...result, prediction: 'Disease_Confounded', visual_stress_indicator: null});
  assert.equal(element('indicatorValue').textContent, '');
  assert.equal(element('indicatorWrap').hidden, true);
  assert.equal(element('indicatorUnavailable').hidden, false);
  assert.equal(element('diagBadge').textContent, 'Disease-like or other confounding symptoms suspected');
  realRenderOutput(result);
  assert.equal(element('indicatorWrap').hidden, false);
  for (const [outcome, status] of [['rejected', 422], ['uncertain', 200], ['model_update_required', 503]]) {
    element('fileInput').files = [{}];
    const body = {outcome, error: outcome === 'uncertain' ? undefined : 'Retake or update model',
      code: outcome === 'rejected' ? 'border_contact' : 'preprocessing_mismatch',
      detail: 'Below the provisional threshold', top_probability: .5996, probability_threshold: .6,
      prediction: null, visual_stress_indicator: null,
      quality: {reason_codes: outcome === 'rejected' ? ['border_contact', 'foreground_too_large'] : []},
      visualizations: {mask: 'data:image/jpeg;base64,mask', candidate_mask: 'data:image/jpeg;base64,candidates'}};
    context.fetch = async () => ({status, ok: status === 200, headers: {get() {return 'application/json';}}, json: async () => body});
    await run('executeScan()');
    assert.equal(element('qualityFeedback').hidden, false);
    assert.equal(element('resultsWrap').style.display, 'none');
    assert.equal(element('consensusBox').style.display, 'none');
    assert.equal(element('qualityMask').src, body.visualizations.mask);
    assert.equal(element('qualityCandidates').src, body.visualizations.candidate_mask);
    assert.equal(element('resetBtn').disabled, false);
    if (outcome === 'uncertain') {
      assert.equal(element('qualityTitle').textContent, 'Uncertain / unsupported input');
      assert.match(element('qualityReasons').textContent, /0.5996/);
    }
    if (outcome === 'rejected') {
      assert.equal(element('qualityTitle').textContent, 'Retake photo');
      assert.match(element('qualityReasons').textContent, /border_contact, foreground_too_large/);
    }
    run('clearScan()');
    assert.equal(element('qualityFeedback').hidden, true);
    assert.equal(element('qualityMask').src, undefined);
  }
  console.log('PASS: Stage 3 mask preview, all rejection reasons, uncertainty, incompatible-model feedback and reset');
  console.log('PASS: Stage 2 integer indicator, disease suppression, descriptive comparisons and disease labels');
  console.log('PASS: frontend stale results, reset lock, URL cleanup, non-JSON/JSON errors, successful completion');
})().catch(error => {console.error(error); process.exitCode = 1;});
