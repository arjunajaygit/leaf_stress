// Optional browser QA: Playwright + installed Edge. Simulated result states are labelled in the report.
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),port=5019;
const output=path.join(root,'reports','presentation_'+new Date().toISOString().replace(/[:.]/g,'-'));
let server,browser,log='';
(async()=>{
  fs.mkdirSync(output,{recursive:true});
  const env={...process.env,PORT:String(port)};delete env.LECOR_MIN_CLASS_PROBABILITY;
  server=spawn(path.join(root,'.venv-py313','Scripts','python.exe'),['run_wholeplant.py'],{cwd:root,env,windowsHide:true,stdio:['ignore','pipe','pipe']});
  server.stdout.on('data',x=>log+=x);server.stderr.on('data',x=>log+=x);
  let ready=false;
  for(let i=0;i<150;i++){try{if((await fetch(`http://127.0.0.1:${port}`)).ok){ready=true;break;}}catch{} await new Promise(r=>setTimeout(r,200));}
  assert.ok(ready,log);
  browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1400,height:1050}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}`);
  await page.setInputFiles('#fileInput',path.join(root,'dataset_wholeplant_v1_20261013','Healthy','healthy_001.JPG'));
  await page.waitForFunction(()=>document.getElementById('imagePreview').naturalWidth>0);
  const realResponse=page.waitForResponse(r=>r.url().endsWith('/predict'));
  await page.click('#scanBtn');const response=await realResponse,actual=await response.json();
  assert.equal(response.status(),200);assert.ok(['classified','uncertain'].includes(actual.outcome));
  assert.equal(actual.probability_threshold,.5);
  await page.screenshot({path:path.join(output,'actual-model.png'),fullPage:true});
  const summaries={Healthy:['Low visible stress','The selected foliage resembles the model’s healthy examples. Early water stress may still be invisible.','Check soil moisture and continue observing the plant.'],
    Moderate_Water_Stress:['Moderate visual stress','The selected foliage resembles the moderate water-stress examples. Other growing conditions can produce similar symptoms.','Check soil moisture and inspect affected leaves before deciding whether to water.'],
    Severe_Water_Stress:['Severe visual stress','The selected foliage resembles the severe water-stress examples. The image alone cannot establish the cause.','Inspect the plant promptly and check soil moisture.'],
    Disease_Confounded:['Other symptoms — water stress unclear','A water-stress level cannot be assigned from this result.','Inspect for pests, disease and nutrient issues.']};
  let sample;
  await page.route('**/predict',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(sample)}));
  for(const [label,[title,explanation,next_check]] of Object.entries(summaries)){
    const probabilities={Healthy:.503,Moderate_Water_Stress:.391,Disease_Confounded:.056,Severe_Water_Stress:.05};
    if(label!=='Healthy'){probabilities.Healthy=probabilities[label];probabilities[label]=.503;}
    sample={outcome:'classified',prediction:label,probability_threshold:.5,probabilities,
      result_summary:{title,explanation,next_check,tentative:true,reliability:'Tentative visual estimate'},best_model_name:'UI test fixture',multi_model_predictions:{},
      visual_stress_indicator:label==='Disease_Confounded'?null:10,advisory:{guidance:explanation,action_items:['Check soil moisture directly.']},
      tissue_stats:{healthy_pct:80,chlorosis_pct:15,necrosis_pct:5},features:{},feature_comparisons:[],visualizations:{},quality:{warnings:[]},model_explanation:{available:false,message:'UI fixture'}};
    await page.click('#scanBtn');await page.waitForFunction(t=>document.getElementById('diagBadge').textContent===t,title);
    assert.equal(await page.locator('#estimateNote').textContent(),'Tentative visual estimate');
    assert.equal(await page.locator('#indicatorValue').isVisible(),false);
    assert.equal(await page.locator('#probabilityBars').isVisible(),false);
    assert.equal(await page.locator('#samCanvas').isVisible(),false);
    if(label==='Healthy'){
      await page.screenshot({path:path.join(output,'tentative-result-desktop.png'),fullPage:true});
      await page.emulateMedia({media:'print'});await page.pdf({path:path.join(output,'sample-report.pdf'),format:'A4',margin:{top:'12mm',bottom:'12mm',left:'12mm',right:'12mm'}});
      await page.emulateMedia({media:'screen'});
      await page.setViewportSize({width:390,height:844});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
      await page.screenshot({path:path.join(output,'tentative-result-mobile.png'),fullPage:true});
      await page.setViewportSize({width:1400,height:1050});
    }
  }
  sample={outcome:'uncertain',detail:'Technical diagnostic text',top_probability:.49,probability_threshold:.5,quality:{warnings:['Long technical warning']},visualizations:{},probabilities:{Healthy:.49,Moderate_Water_Stress:.48}};
  await page.click('#scanBtn');await page.waitForFunction(()=>!document.getElementById('qualityFeedback').hidden);
  assert.ok(!(await page.locator('#qualityMessage').textContent()).includes('Technical'));
  assert.deepEqual(errors,[]);
  const report={real_model_outcome:actual.outcome,real_threshold:actual.probability_threshold,
    simulated_UI_states:Object.keys(summaries),tentative_label_visible:true,technical_details_collapsed:true,mobile_overflow:false,print_report_generated:true,page_errors:errors,
    note:'Category screenshots use explicitly simulated UI states; actual-model.png uses the current saved model. This checks UI behaviour, not accuracy.'};
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log('PASS: actual upload, four UI categories, tentative label, collapsed details, mobile and print');console.log(output);
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();if(server)server.kill();fs.writeFileSync(path.join(output,'server.log'),log);});
