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
  console.log('PASS: frontend stale results, reset lock, URL cleanup, non-JSON/JSON errors, successful completion');
})().catch(error => {console.error(error); process.exitCode = 1;});
