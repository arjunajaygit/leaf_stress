/* Point-guided local SAM, with explicit mask review before classifier input changes. */
window.plantSam = (() => {
  let points = [], source = null, overlay = null, result = null, prepared = null;
  let generation = 0, approved = false;
  let selectionRecord = null;
  const el = id => document.getElementById(id);
  const busy = () => typeof scanInFlight !== 'undefined' && scanInFlight;
  const say = text => { el('samStatus').textContent = text; };
  function discard() {
    result = prepared = overlay = selectionRecord = null; approved = false;
    el('samResult').hidden = true; el('samApply').disabled = true;
    el('samIsolated').removeAttribute('src');
    clearResults();
  }
  function reset() {
    generation++; points = []; source = null; discard();
    const c = el('samCanvas'); c.getContext('2d').clearRect(0,0,c.width,c.height);
    say('Mark the target plant with Keep, then mark background with Exclude.');
  }
  function draw() {
    if (!source) return;
    const c = el('samCanvas'), ctx = c.getContext('2d');
    ctx.drawImage(overlay || source,0,0,c.width,c.height);
    points.forEach((p,i) => {
      const x = p.x*c.width, y = p.y*c.height;
      ctx.beginPath(); ctx.arc(x,y,6,0,2*Math.PI);
      ctx.fillStyle = p.label ? '#10ff80' : '#ff4050'; ctx.fill();
      ctx.strokeStyle = '#000'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = '14px sans-serif'; ctx.fillText(String(i+1),x+9,y);
    });
  }
  async function loadPreview() {
    const current = el('imagePreview');
    if (!current.src || !current.naturalWidth) return;
    reset();
    const token = generation, image = new Image(); image.src = current.src;
    try {
      await image.decode(); if (token !== generation) return;
      source = image;
      const c = el('samCanvas'), scale = Math.min(1,720/image.naturalWidth);
      c.width = Math.max(1,Math.round(image.naturalWidth*scale)); c.height = Math.max(1,Math.round(image.naturalHeight*scale)); draw();
    } catch { say('Could not display the photo for SAM. Choose another photo.'); }
  }
  function add(x,y) {
    if (busy() || !source) return;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x<0 || x>1 || y<0 || y>1) { say('Enter X and Y between 0 and 100%.'); return; }
    if (points.length >= 64) { say('Maximum 64 marks. Undo or clear a mark first.'); return; }
    points.push({x,y,label:Number(el('samMark').value)});
    generation++; discard(); draw(); say(`${points.length} marks. Generate the SAM mask when ready.`);
  }
  function setBusy(value) {
    scanInFlight = value;
    for (const id of ['scanBtn','resetBtn','fileInput','samRun','samAdd','samUndo','samClear','applyCrop','fullCrop','samMark','samX','samY','cropLeft','cropTop','cropRight','cropBottom']) el(id).disabled = value;
  }
  function fromDataURL(url, name) {
    if (typeof url !== 'string' || !url.startsWith('data:image/png;base64,')) throw new Error('SAM returned invalid image data.');
    const bytes = Uint8Array.from(atob(url.split(',')[1]), char => char.charCodeAt(0));
    return new File([bytes],name,{type:'image/png'});
  }
  async function run() {
    if (busy() || !source) return;
    if (window.plantCrop?.pending) { say('Apply the crop first; then add SAM marks on the applied crop.'); return; }
    if (!points.some(p => p.label===1)) { say('Add at least one Keep mark on the target plant.'); return; }
    const file = window.plantCrop?.file || fileInput.files[0], token = generation;
    if (!file) return;
    discard(); setBusy(true); say('SAM is processing locally. First use loads the model; CPU inference can take time…');
    const body = new FormData(); body.append('image',file); body.append('points',JSON.stringify(points));
    try {
      const response = await fetch('/api/sam/segment',{method:'POST',body});
      if (!(response.headers.get('content-type') || '').includes('application/json')) throw new Error(`Unreadable SAM response (HTTP ${response.status}).`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'SAM failed.');
      const next = fromDataURL(data.image,'target-plant-sam.png');
      const image = new Image(); image.src = data.overlay; await image.decode();
      if (token !== generation) return;
      result = data; prepared = next; overlay = image;
      selectionRecord = {model:data.model,source_filename:fileInput.files[0]?.name,
        applied_crop_percent:window.plantCrop?.bounds || [0,0,100,100],points:points.map(p=>({...p})),
        output_width:data.width,output_height:data.height,background:220,
        notes:'Coordinates are normalized to the applied crop. Record genuine plant/session IDs separately.'};
      el('samIsolated').src = data.image; el('samResult').hidden = false;
      el('samApply').disabled = false; draw();
      say('Review the overlay for unwanted trees and missing leaves. Add marks and regenerate if needed, then Use reviewed plant region.');
    } catch (error) { if (token===generation) say(error.message); }
    finally { setBusy(false); }
  }
  function download(file) {
    const url = URL.createObjectURL(file), link = document.createElement('a');
    link.href = url; link.download = file.name; link.click();
    setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  document.addEventListener('DOMContentLoaded', () => {
    el('imagePreview').addEventListener('load',loadPreview);
    el('samCanvas').addEventListener('click',event => {
      const r = event.currentTarget.getBoundingClientRect();
      add(Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(event.clientY-r.top)/r.height)));
    });
    el('samAdd').addEventListener('click', () => add(Number(el('samX').value)/100,Number(el('samY').value)/100));
    el('samUndo').addEventListener('click', () => { if (busy()) return; points.pop(); generation++; discard(); draw(); say(`${points.length} marks. Generate the mask again.`); });
    el('samClear').addEventListener('click', () => { if (!busy()) loadPreview(); });
    el('samRun').addEventListener('click',run);
    el('samApply').addEventListener('click', () => {
      if (!prepared || busy()) return;
      approved = true; clearResults();
      say('Reviewed SAM region applied. Preview foliage mask, then Analyze image. Grey background will be excluded.');
    });
    el('samDownload').addEventListener('click', () => { if (prepared && !busy()) download(prepared); });
    el('samDownloadMask').addEventListener('click', () => { if (result && !busy()) download(fromDataURL(result.mask,'target-plant-sam-mask.png')); });
    el('samDownloadMarks').addEventListener('click', () => {
      if (!result || busy()) return;
      download(new File([JSON.stringify(selectionRecord,null,2)],'target-plant-sam-selection.json',{type:'application/json'}));
    });
    fetch('/api/sam/status').then(response => response.json()).then(data => {
      el('samAvailability').textContent = data.message || data.error || 'SAM status unavailable.';
    }).catch(() => { el('samAvailability').textContent = 'Could not check SAM setup. Check the server connection.'; });
  });
  return {reset, get pending() { return points.length > 0 && !approved; },
    get file() { return approved ? prepared : null; }, get preview() { return approved ? result?.image : null; }};
})();
