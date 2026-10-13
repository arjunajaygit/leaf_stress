// Verify review-before-analysis, prompt edits, request inputs and failure recovery.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
const context2d = {clearRect(){},drawImage(){},beginPath(){},arc(){},fill(){},stroke(){},fillText(){}};
function el(id) {
  if (!elements.has(id)) elements.set(id, {
    value:'', disabled:false, hidden:false, width:150, height:100, listeners:{},
    addEventListener(name,fn){this.listeners[name]=fn;},
    getContext(){return context2d;},removeAttribute(name){delete this[name];},
    getBoundingClientRect(){return {left:0,top:0,width:150,height:100};},click(){},
  });
  return elements.get(id);
}
let ready, fail = false, request, clears = 0;
const url = 'data:image/png;base64,' + Buffer.from('test-png').toString('base64');
const crop = new File(['crop'],'crop.png',{type:'image/png'});
el('imagePreview').src='blob:source'; el('imagePreview').naturalWidth=150;
el('samMark').value='1'; el('samX').value='30'; el('samY').value='40';
const context = vm.createContext({
  window:{plantCrop:{file:crop,pending:false}},
  document:{getElementById:el,addEventListener(name,fn){if(name==='DOMContentLoaded')ready=fn;},createElement(){return el('download');}},
  Image:class {naturalWidth=150;naturalHeight=100;async decode(){}},
  File, Uint8Array, atob, FormData, URL, setTimeout(){},
  scanInFlight:false,fileInput:{files:[new File(['original'],'original.jpg')]},
  clearResults(){clears++;},
  async fetch(path,options){
    if (path.endsWith('/status')) return {json:async()=>({message:'ready'})};
    request=options.body;
    return {ok:!fail,status:fail?503:200,headers:{get(){return 'application/json';}},json:async()=>fail?
      {error:'SAM setup required'}:{image:url,mask:url,overlay:url,width:150,height:100,model:'SAM ViT-B'}};
  },
});
vm.runInContext(fs.readFileSync('static/plant_sam.js','utf8'),context);
const sam = context.window.plantSam;
(async()=>{
  ready(); await el('imagePreview').listeners.load();
  el('samAdd').listeners.click(); assert.equal(sam.pending,true); assert.equal(sam.file,null);
  await el('samRun').listeners.click();
  assert.equal(request.get('image').name,'crop.png');
  assert.deepEqual(JSON.parse(request.get('points')),[{x:.3,y:.4,label:1}]);
  assert.equal(sam.file,null,'generation alone must not apply the mask');
  el('samApply').listeners.click();
  assert.equal(sam.file.name,'target-plant-sam.png'); assert.equal(sam.pending,false);
  el('samMark').value='0'; el('samX').value='90';
  el('samAdd').listeners.click();
  assert.equal(sam.file,null,'new marks must invalidate the approved mask'); assert.equal(sam.pending,true);
  fail=true; await el('samRun').listeners.click();
  assert.equal(sam.file,null); assert.equal(context.scanInFlight,false);
  assert.equal(el('samStatus').textContent,'SAM setup required');
  await el('samClear').listeners.click();
  assert.equal(sam.pending,false); assert.equal(sam.file,null); assert.ok(clears>0);
  console.log('PASS: SAM review, crop request, changed marks, error recovery and clear');
})().catch(error=>{console.error(error);process.exitCode=1;});
