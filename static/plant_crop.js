/* Crop original pixels; the same exported PNG can be used for training. */
window.plantCrop = (() => {
  let source = null, selected = null, generation = 0, start = null, dirty = false;
  let appliedBounds = [0,0,100,100];
  const ids = ['cropLeft', 'cropTop', 'cropRight', 'cropBottom'];
  const el = id => document.getElementById(id);
  const busy = () => typeof scanInFlight !== 'undefined' && scanInFlight;
  const status = message => { el('cropStatus').textContent = message; };
  function reset() {
    generation++; source = selected = start = null; dirty = false;
    appliedBounds = [0,0,100,100];
  }
  function bounds() {
    const b = ids.map(id => Number(el(id).value));
    if (!b.every(Number.isFinite) || b.some(n => n < 0 || n > 100) || b[2] <= b[0] || b[3] <= b[1])
      throw new Error('Choose a rectangle with left < right and top < bottom, between 0 and 100%.');
    return b;
  }
  function draw() {
    if (!source) return;
    const canvas = el('cropCanvas'), ctx = canvas.getContext('2d');
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    try {
      const [l,t,r,b] = bounds();
      ctx.strokeStyle = '#10b981'; ctx.lineWidth = 3;
      ctx.strokeRect(l*canvas.width/100, t*canvas.height/100, (r-l)*canvas.width/100, (b-t)*canvas.height/100);
    } catch { /* Invalid typed bounds are explained when applying. */ }
  }
  async function load(file) {
    const token = generation, url = URL.createObjectURL(file), image = new Image();
    try {
      image.src = url; await image.decode();
      if (token !== generation) return;
      if (image.naturalWidth * image.naturalHeight > 16000000) throw new Error('Use an image with at most 16 million pixels.');
      source = image;
      const canvas = el('cropCanvas'), scale = Math.min(1, 720/image.naturalWidth);
      canvas.width = Math.round(image.naturalWidth*scale); canvas.height = Math.round(image.naturalHeight*scale);
      ids.forEach((id,i) => el(id).value = i < 2 ? 0 : 100);
      draw(); status('Full photo selected. Adjust the rectangle, apply it, then preview the mask.');
    } catch (error) { status(error.message); }
    finally { URL.revokeObjectURL(url); }
  }
  async function apply() {
    if (!source || busy()) return;
    try {
      const [l,t,r,b] = bounds(), token = generation;
      const x = Math.floor(l*source.naturalWidth/100), y = Math.floor(t*source.naturalHeight/100);
      const w = Math.ceil(r*source.naturalWidth/100)-x, h = Math.ceil(b*source.naturalHeight/100)-y;
      if (Math.min(w,h) < 32) throw new Error('Select an area at least 32 pixels wide and high.');
      const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
      canvas.getContext('2d').drawImage(source,x,y,w,h,0,0,w,h);
      const blob = await new Promise(resolve => canvas.toBlob(resolve,'image/png'));
      if (!blob) throw new Error('Could not create the crop.');
      if (token !== generation || busy()) return;
      if (blob.size > 9*1024*1024) throw new Error('Crop is too large to upload; choose a smaller photo.');
      selected = new File([blob], 'target-plant.png', {type:'image/png'});
      appliedBounds = [l,t,r,b];
      dirty = false;
      window.plantSam?.reset();
      clearResults(); releasePreview();
      previewUrl = URL.createObjectURL(selected); imagePreview.src = previewUrl;
      status(`Applied ${w} × ${h} crop. Preview the mask; overlapping foliage can remain.`);
    } catch (error) { status(error.message); }
  }
  document.addEventListener('DOMContentLoaded', () => {
    const canvas = el('cropCanvas');
    const point = event => {
      const rect = canvas.getBoundingClientRect();
      return [Math.max(0,Math.min(100,(event.clientX-rect.left)/rect.width*100)), Math.max(0,Math.min(100,(event.clientY-rect.top)/rect.height*100))];
    };
    canvas.addEventListener('pointerdown', event => {
      if (!source || busy()) return;
      start = point(event); canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!start || busy()) return;
      const end = point(event), values = [Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];
      ids.forEach((id,i) => el(id).value = values[i].toFixed(2)); draw();
      dirty = true; status('Selection changed. Apply crop to use it.');
    });
    for (const name of ['pointerup','pointercancel']) canvas.addEventListener(name, () => { start = null; });
    ids.forEach(id => el(id).addEventListener('input', () => { if (busy()) return; dirty = true; draw(); status('Selection changed. Apply crop to use it.'); }));
    el('applyCrop').addEventListener('click', apply);
    el('fullCrop').addEventListener('click', () => {
      if (busy()) return;
      ids.forEach((id,i) => el(id).value = i < 2 ? 0 : 100); draw(); apply();
    });
    el('saveCrop').addEventListener('click', () => {
      if (!selected || dirty || busy()) { status('Apply your crop before downloading.'); return; }
      const url = URL.createObjectURL(selected), link = document.createElement('a');
      link.href = url; link.download = 'target-plant.png'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
  });
  return {reset, load, get file() { return selected; }, get pending() { return dirty; }, get bounds() { return appliedBounds.slice(); }};
})();
