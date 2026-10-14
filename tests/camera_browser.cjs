// Playwright + Edge QA using a simulated camera only; never accesses a physical webcam.
const {chromium}=require('playwright'),{spawn}=require('node:child_process');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),port=5023;
const output=path.join(root,'reports','camera_'+new Date().toISOString().replace(/[:.]/g,'-'));
let browser,server,log='';
(async()=>{
  fs.mkdirSync(output,{recursive:true});
  server=spawn(path.join(root,'.venv-py313','Scripts','python.exe'),['run_wholeplant.py'],{cwd:root,env:{...process.env,PORT:String(port)},windowsHide:true,stdio:['ignore','pipe','pipe']});
  server.stdout.on('data',x=>log+=x);server.stderr.on('data',x=>log+=x);
  let ready=false;
  for(let i=0;i<150;i++){try{if((await fetch(`http://127.0.0.1:${port}`)).ok){ready=true;break;}}catch{}await new Promise(r=>setTimeout(r,200));}
  assert.ok(ready,log);
  browser=await chromium.launch({channel:'msedge',headless:true,args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
  const page=await browser.newPage({viewport:{width:1280,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}`);
  await page.evaluate(()=>{
    const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    window.testGetMedia=original;
    navigator.mediaDevices.getUserMedia=async constraints=>{window.testConstraints=constraints;window.testStream=await original(constraints);return window.testStream;};
  });
  await page.click('#openCamera');
  await page.waitForFunction(()=>!document.getElementById('capturePhoto').disabled);
  assert.equal(await page.evaluate(()=>window.testConstraints.audio),false);
  await page.click('#capturePhoto');await page.waitForFunction(()=>!document.getElementById('usePhoto').hidden);
  assert.equal(await page.evaluate(()=>window.testStream.getTracks().every(t=>t.readyState==='ended')),true);
  assert.equal(await page.locator('#cameraReview').isVisible(),true);
  await page.click('#retakePhoto');await page.waitForFunction(()=>!document.getElementById('capturePhoto').disabled);
  await page.click('#capturePhoto');await page.waitForFunction(()=>!document.getElementById('usePhoto').hidden);
  await page.click('#usePhoto');
  await page.waitForFunction(()=>document.getElementById('imagePreview').naturalWidth>0);
  const selected=await page.evaluate(()=>({name:document.getElementById('fileInput').files[0].name,type:document.getElementById('fileInput').files[0].type,
    width:document.getElementById('imagePreview').naturalWidth,height:document.getElementById('imagePreview').naturalHeight,panelHidden:document.getElementById('cameraPanel').hidden}));
  assert.equal(selected.name,'plant-camera.jpg');assert.equal(selected.type,'image/jpeg');assert.ok(Math.max(selected.width,selected.height)<=1600);assert.equal(selected.panelHidden,true);
  const predicted=page.waitForResponse(r=>r.url().endsWith('/predict'));
  await page.click('#scanBtn');const response=await predicted,data=await response.json();
  assert.ok([200,422].includes(response.status()),JSON.stringify(data));assert.ok(['classified','uncertain','rejected'].includes(data.outcome));
  await page.screenshot({path:path.join(output,'captured-photo.png'),fullPage:true});
  // A denied permission is handled without affecting the selected image.
  await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=()=>Promise.reject(new DOMException('Denied','NotAllowedError'));});
  await page.click('#openCamera');await page.waitForFunction(()=>document.getElementById('cameraStatus').textContent.includes('denied'));
  assert.equal(await page.locator('#cameraPanel').isVisible(),false);
  assert.equal(await page.evaluate(()=>document.getElementById('fileInput').files[0].name),'plant-camera.jpg');
  // Cancel while permission is pending; a late stream must immediately be stopped.
  await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>window.resolveTestCamera=resolve);});
  await page.click('#openCamera');await page.click('#closeCamera');
  await page.evaluate(()=>{const canvas=document.createElement('canvas');canvas.width=100;canvas.height=100;window.lateStream=canvas.captureStream(10);window.resolveTestCamera(window.lateStream);});
  await page.waitForFunction(()=>window.lateStream.getTracks().every(t=>t.readyState==='ended'));
  // Device picker uses the same accepted-file path.
  await page.setInputFiles('#cameraFileInput',path.join(root,'dataset_wholeplant_v1_20261013','Healthy','healthy_001.JPG'));
  assert.equal(await page.evaluate(()=>document.getElementById('fileInput').files[0].name),'healthy_001.JPG');
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({simulated_camera:true,selected,api_status:response.status(),outcome:data.outcome,
    capture_review_retake:true,no_audio:true,tracks_stopped:true,permission_denial_handled:true,late_cancel_cleanup:true,device_picker_path:true,page_errors:errors,
    limitation:'Uses browser-simulated video, not a physical webcam. Real mobile device/camera permissions still need manual verification.'},null,2));
  console.log('PASS: simulated camera, capture/review/retake, real analysis upload, denial and stream cleanup');console.log(output);
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();if(server)server.kill();fs.writeFileSync(path.join(output,'server.log'),log);});
