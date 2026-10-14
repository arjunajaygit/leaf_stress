/* User-initiated still-photo capture. No microphone and no automatic upload. */
window.plantCamera=(()=>{
  let stream=null,photo=null,reviewURL=null,generation=0,facing='environment';
  const el=id=>document.getElementById(id);
  const busy=()=>typeof scanInFlight!=='undefined'&&scanInFlight;
  const say=text=>{el('cameraStatus').textContent=text;};
  function stop(){
    if(stream)stream.getTracks().forEach(track=>track.stop());
    stream=null;el('cameraVideo').srcObject=null;el('capturePhoto').disabled=true;
  }
  function clearReview(){
    photo=null;if(reviewURL)URL.revokeObjectURL(reviewURL);reviewURL=null;
    el('cameraReview').removeAttribute('src');el('cameraReview').hidden=true;
  }
  function close(){
    generation++;stop();clearReview();el('cameraPanel').hidden=true;
    el('openCamera').disabled=false;say('');
  }
  async function open(){
    if(busy())return;
    close();const token=generation;
    if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){
      say('Live camera needs HTTPS or localhost. You can use Device camera / photo picker, or upload a photo.');return;
    }
    el('cameraPanel').hidden=false;el('cameraVideo').hidden=false;
    el('usePhoto').hidden=el('retakePhoto').hidden=true;
    el('capturePhoto').hidden=el('switchCamera').hidden=false;
    el('openCamera').disabled=true;say('Allow camera access in your browser to continue.');
    try{
      const next=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing},width:{ideal:1600},height:{ideal:1200}},audio:false});
      if(token!==generation){next.getTracks().forEach(track=>track.stop());return;}
      stream=next;el('cameraVideo').srcObject=stream;
      await el('cameraVideo').play();
      if(token!==generation)return;
      el('capturePhoto').disabled=el('cameraVideo').videoWidth===0;
      say('Camera ready. Keep the plant steady and capture a clear photo.');
    }catch(error){
      if(token!==generation)return;
      stop();el('cameraPanel').hidden=true;
      const messages={NotAllowedError:'Camera permission was denied. Allow it in browser settings, or upload a photo.',
        NotFoundError:'No camera was found. Connect a webcam or use the device photo picker.',
        NotReadableError:'The camera is busy or unavailable. Close other camera apps and try again.'};
      say(messages[error.name]||'Could not open the camera. Try the device photo picker or upload a photo.');
    }finally{if(token===generation)el('openCamera').disabled=false;}
  }
  async function capture(){
    if(busy()||!stream||el('capturePhoto').disabled)return;
    const video=el('cameraVideo'),token=generation,w=video.videoWidth,h=video.videoHeight;
    if(!w||!h){say('Wait for the camera preview, then try again.');return;}
    el('capturePhoto').disabled=true;
    const scale=Math.min(1,1600/Math.max(w,h)),canvas=document.createElement('canvas');
    canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));
    try{
      canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
      const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.92));
      if(token!==generation)return;
      if(!blob||blob.size>9*1024*1024)throw new Error('Could not prepare the photo. Please retake it.');
      photo=new File([blob],'plant-camera.jpg',{type:'image/jpeg'});stop();
      reviewURL=URL.createObjectURL(photo);el('cameraReview').src=reviewURL;el('cameraReview').hidden=false;
      el('cameraVideo').hidden=true;el('capturePhoto').hidden=el('switchCamera').hidden=true;
      el('usePhoto').hidden=el('retakePhoto').hidden=false;
      say('Review your photo. Use it to continue, or retake it. The camera is now off.');
    }catch(error){if(token===generation){el('capturePhoto').disabled=false;say(error.message);}}
  }
  function use(file){
    if(busy()||!file)return;
    if(!/\.(jpe?g|png|webp)$/i.test(file.name)||(file.type&&!['image/jpeg','image/png','image/webp'].includes(file.type))){say('Use a JPEG, PNG or WebP photo. Convert HEIC images before uploading.');return;}
    try{
      const transfer=new DataTransfer();transfer.items.add(file);fileInput.files=transfer.files;
      showSelectedFile();say('Photo selected. Crop or refine the plant if needed, then choose Analyze image.');
    }catch{say('This browser could not select the captured photo. Use the device photo picker or file upload.');}
  }
  document.addEventListener('DOMContentLoaded',()=>{
    el('openCamera').addEventListener('click',open);el('closeCamera').addEventListener('click',close);
    el('capturePhoto').addEventListener('click',capture);el('retakePhoto').addEventListener('click',open);
    el('switchCamera').addEventListener('click',()=>{if(!busy()){facing=facing==='environment'?'user':'environment';open();}});
    el('usePhoto').addEventListener('click',()=>use(photo));
    el('nativeCamera').addEventListener('click',()=>{if(!busy()){close();el('cameraFileInput').value='';el('cameraFileInput').click();}});
    el('cameraFileInput').addEventListener('change',()=>use(el('cameraFileInput').files?.[0]));
    el('cameraVideo').addEventListener('loadedmetadata',()=>{if(stream)el('capturePhoto').disabled=false;});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)close();});
    window.addEventListener('pagehide',close);
  });
  return {close};
})();
