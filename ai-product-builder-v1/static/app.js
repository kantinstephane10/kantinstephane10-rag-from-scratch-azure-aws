'use strict';
const $=id=>document.getElementById(id);
const state={roles:{},layers:[],meetings:[],projects:[],selected:null,room:null,demo:false,renderedIds:'',view:'architecture',selectedTaskIds:new Set()};
const nice=id=>id.replaceAll('-',' ').replace(/\b\w/g,c=>c.toUpperCase());
const time=t=>t?new Date(t*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}):'';
const mk=(tag,attrs={},text)=>{const e=document.createElement(tag);for(const [k,v] of Object.entries(attrs)){if(k==='class')e.className=v;else if(k==='value')e.value=v;else e.setAttribute(k,v)}if(text!==undefined)e.textContent=text;return e;};
async function api(url,options){const r=await fetch(url,{cache:'no-store',...options});let data={};try{data=await r.json()}catch{throw Error('Server returned an invalid response.')}if(!r.ok)throw Error(data.error||`HTTP ${r.status}`);return data;}
async function post(url,obj={}){return api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});}
function alertMessage(s){const a=$('alert');a.hidden=!s;a.textContent=s||'';if(s)setTimeout(()=>{if(a.textContent===s)a.hidden=true;},7000);}
function switchView(view){
 state.view=view;
 for(const v of ['meetings','architecture','projects','handoffs','settings'])$('view-'+v).hidden=v!==view;
 document.querySelectorAll('.nav').forEach(n=>n.classList.toggle('active',n.dataset.view===view));
 $('view-name').textContent={meetings:'Meeting studio',architecture:'Organization',projects:'Projects',handoffs:'Handoffs & artifacts',settings:'OpenAI & model settings'}[view];
 document.body.classList.toggle('architecture-mode',view==='architecture');
 document.body.classList.remove('navigation-open');$('navigation-backdrop').hidden=true;
 if(view==='handoffs')refreshEvidence().catch(e=>alertMessage(e.message));
 if(view==='settings')refreshSettings().catch(e=>alertMessage(e.message));
}

function fillSelect(select,entries,first){const old=select.value;select.replaceChildren();if(first)select.append(mk('option',{value:first[0]},first[1]));entries.forEach(([value,label])=>select.append(mk('option',{value},label)));if([...select.options].some(o=>o.value===old))select.value=old;}
const FLOW_ICON_MAP={
 'business-analyst-1':'chart-no-axes-column','business-analyst-2':'chart-no-axes-column','product-manager':'briefcase-business',
 'requirements-validator':'clipboard-check','cost-optimization-agent':'coins','solution-architect':'network','orchestrator':'bot',
 'project-manager':'calendar-days','dev-manager':'settings-2','rag-engineer':'database',
 'agentic-ai-engineer':'workflow','llm-engineer':'cpu','memory-integration-engineer':'brain-circuit',
 'data-engineer':'database','prompt-engineer':'terminal','tool-integration-engineer':'plug','performance-engineer':'gauge',
 'security-agent':'shield-check','evaluation-agent':'flask-conical','red-team-agent':'shield-alert',
 'senior-troubleshooter':'cog','tester-1':'test-tube-diagonal','tester-2':'test-tube-diagonal',
 'code-approval-manager-1':'user-check','code-approval-manager-2':'user-check','delegated-approval-agent':'user-check',
 'release-manager':'rocket','devops-mlops-agent':'infinity','monitoring-agent':'activity'};
const FLOW_LABEL_MAP={
 'cost-optimization-agent':'Cost Estimator','orchestrator':'AI Orchestrator','dev-manager':'Development Manager',
 'memory-integration-engineer':'Memory Engineer','tool-integration-engineer':'Tool Integration',
 'security-agent':'AI Security','evaluation-agent':'AI Evaluation','red-team-agent':'AI Red Team',
 'code-approval-manager-1':'Code Approval 1','code-approval-manager-2':'Code Approval 2',
 'delegated-approval-agent':'Delegated Approval','devops-mlops-agent':'DevOps / MLOps'};
const FLOW_LAYERS={L1:'Strategy & product',L2:'Architecture & orchestration',L3:'Development management',
 L4:'Specialized engineering',L5:'Development & review',L6:'Quality, security & evaluation',
 L7:'Testing & validation',L8:'Approval & delivery',L9:'Operations & improvement'};
const flowName=id=>{if(FLOW_LABEL_MAP[id])return FLOW_LABEL_MAP[id];
 if(id.startsWith('dev-'))return 'Developer '+id.split('-')[1];
 if(id.startsWith('code-review-'))return 'Reviewer '+id.split('-')[2];
 if(id.startsWith('lead-tester-'))return 'Lead Tester '+id.split('-')[2];
 if(id.startsWith('tester-'))return 'Tester '+id.split('-')[1];
 return nice(id);
};
function iconFor(id, cssClass='flow-role-icon'){
 const el=document.createElementNS('http://www.w3.org/2000/svg','svg');
 el.setAttribute('class',cssClass); el.setAttribute('aria-hidden','true');
 const use=document.createElementNS('http://www.w3.org/2000/svg','use');
 use.setAttribute('href','/icons.svg#'+(FLOW_ICON_MAP[id] || (id.startsWith('dev-')?'code-xml':id.startsWith('code-review-')?'file-search':id.startsWith('lead-tester-')?'shield-check':'activity')));
 el.append(use);return el;
}
function flowTask(role){const rows=state.activity?.messages||[];return [...rows].reverse().find(x=>x.recipient===role)}
function flowStatus(role){const task=flowTask(role);return task?task.status:'queued'}
function roleTile(id){
 const task=flowTask(id),status=flowStatus(id),label=flowName(id);
 const el=mk('button',{type:'button',class:'flow-role state-'+status,'data-role':id,
  title:`${label} — ${task?'Task '+status:'No assigned task yet'}. ${state.roles[id]||''}`});
 el.append(iconFor(id),mk('span',{class:'flow-role-name'},label));
 if(status==='done')el.append(mk('span',{class:'flow-role-status'},'✓'));
 else if(status==='processing')el.append(mk('span',{class:'flow-role-status'},'●'));
 else if(status==='failed'||status==='blocked'||status==='cancelled')el.append(mk('span',{class:'flow-role-status'},'!'));
 el.addEventListener('click',async()=>{if(!state.projectSelected){switchView('projects');$('project-name').focus();return;}
  if(task){state.selectedTask=task.id;switchView('projects');await refreshActivity();$('project-live').scrollIntoView({behavior:'smooth',block:'start'});}
  else{switchView('projects');alertMessage(`${label} is registered but has no assigned task in this project yet.`)}
 });return el;
}
function flowGrid(ids,css=''){
 const grid=mk('div',{class:'flow-role-grid '+css});
 for(const id of ids)grid.append(roleTile(id));
 return grid;
}
function renderArchitecture(){
 const root=$('architecture-layers');root.replaceChildren();
 for(const layer of state.layers){
  const wrap=mk('div',{class:'flow-layer','data-layer':layer.id});
  const lbl=mk('div',{class:'flow-layer-label'});
  lbl.append(mk('span',{class:'flow-layer-id'},layer.id),mk('b',{},FLOW_LAYERS[layer.id]||layer.name));
  const content=mk('div',{class:'flow-layer-contents'});
  if(layer.id==='L5'){
   content.append(flowGrid(layer.roles.filter(x=>x.startsWith('dev-')),'flow-developers'));
   content.append(flowGrid(layer.roles.filter(x=>x.startsWith('code-review-')),'flow-reviewers'));
  } else if(layer.id==='L7'){
   content.append(flowGrid(layer.roles.filter(x=>x.startsWith('tester-')),'flow-testers'));
   content.append(flowGrid(layer.roles.filter(x=>x.startsWith('lead-tester-')),'flow-lead-testers'));
  } else if(layer.id==='L8'){
   const ids=layer.roles;
   const arranged=['code-approval-manager-1','code-approval-manager-2','delegated-approval-agent',
    '@human','release-manager','devops-mlops-agent'].filter(x=>x==='@human'||ids.includes(x));
   const grid=mk('div',{class:'flow-role-grid flow-approvals'});
   for(const role of arranged){
    if(role==='@human'){
     const approved=state.activity?.project?.status==='approved';
     const human=mk('button',{type:'button',class:'flow-role state-human'+(approved?' is-approved':'')});
     human.append(iconFor('delegated-approval-agent'),mk('span',{class:'flow-role-name'},'Human Approval'));
     if(approved)human.append(mk('span',{class:'flow-role-status'},'✓'));
     human.addEventListener('click',()=>{switchView('projects');alertMessage('The human operator must approve protected changes. AI approval is advisory.');});
     grid.append(human);
    }else grid.append(roleTile(role));
   }
   content.append(grid);
  }else content.append(flowGrid(layer.roles,layer.id==='L3'||layer.id==='L9'?'flow-single':''));
  wrap.append(lbl,content);root.append(wrap);
 }
 const data=state.activity;
 const count=data?.counts||{done:0,total:0,processing:0,queued:0};
 const status=data?.project?.status||'draft';
 const badge=$('workflow-badge');
 badge.className='workflow-badge';
 if(!data){badge.textContent='READY';$('workflow-subtitle').textContent='Create a project to start the agent workflow';}
 else if(status==='approved'){badge.textContent='APPROVED';$('workflow-subtitle').textContent=data.project.name+' · Human-approved project';badge.classList.add('success');}
 else if(status==='awaiting_human'&&count.processing>0){badge.textContent='WORKING';$('workflow-subtitle').textContent=data.project.name+' · Advisory agents working; human review pending';badge.classList.add('working');}
 else if(status==='awaiting_human'&&count.queued>0){badge.textContent='QUEUED';$('workflow-subtitle').textContent=data.project.name+' · Specialists scheduled automatically; human review pending';badge.classList.add('queued');}
 else if(status==='awaiting_human'){badge.textContent='REVIEW';$('workflow-subtitle').textContent=data.project.name+' · Advisory work complete; awaiting human decision';badge.classList.add('paused');}
 else if(count.processing>0){badge.textContent='WORKING';$('workflow-subtitle').textContent=data.project.name+' · Agents processing tasks';badge.classList.add('working');}
 else if(count.total>0&&count.queued===0){badge.textContent='PAUSED';$('workflow-subtitle').textContent=data.project.name+' · Between agent turns';badge.classList.add('paused');}
 else if(count.total>0){badge.textContent='QUEUED';$('workflow-subtitle').textContent=data.project.name+' · Agent tasks queued';badge.classList.add('queued');}
 else{badge.textContent='READY';$('workflow-subtitle').textContent=data.project.name+' · No agent tasks yet';}
 $('workflow-tally').textContent=`${count.done||0} of ${count.total||0} agent tasks recorded`;
}
function renderModelRouting(data){
 const groups=['architecture','engineering','business','routine'];
 for(const group of groups){if($('model-'+group))$('model-'+group).textContent=data?.models?.[group]||'Not configured';}
}
function refreshProjectPicker(){
 const p=$('workflow-project-picker'),selected=state.projectSelected||'';
 p.replaceChildren(mk('option',{value:''},'Select project'));
 for(const item of state.projects)p.append(mk('option',{value:item.id},item.name));
 p.value=selected;
}
function renderProjects(){
 const root=$('project-list');root.replaceChildren();
 if(!state.projects.length)root.append(mk('p',{class:'muted'},'No projects yet.'));
 for(const project of state.projects){
  const entry=mk('div',{class:'project-ledger-entry'});
  const item=mk('button',{class:'project-item'+(state.projectSelected===project.id?' selected':''),type:'button'});
  item.append(mk('b',{},project.name),mk('small',{},project.status+' · '+project.id));
  item.addEventListener('click',async()=>{state.projectSelected=project.id;state.selectedTask=null;state.selectedTaskIds.clear();renderProjects();await refreshActivity();await refreshRagStatus();});
  const del=mk('button',{class:'danger project-delete-button',type:'button','aria-label':'Delete project '+project.name},'Delete');
  del.addEventListener('click',()=>openDeleteDialog(project));
  entry.append(item,del);root.append(entry);
 }
 fillSelect($('meeting-project'),state.projects.map(p=>[p.id,p.name]),['','Standalone meeting']);
}

async function refreshProjects(){
 state.projects=(await api('/api/projects')).projects;
 if(!state.projects.some(p=>p.id===state.projectSelected))state.projectSelected=state.projects[0]?.id||null;
 renderProjects();refreshProjectPicker();
 if(state.projectSelected)await refreshActivity();else {state.activity=null;$('project-live').hidden=true;renderArchitecture();}
 await refreshRagStatus();
}

function intakePayload(){const num=id=>Number($(id).value||'0');return {product_type:$('project-type').value,model_profile:$('model-profile').value,provider_consent:$('provider-consent').checked,estimated_hours:num('estimated-hours'),hourly_rate:num('hourly-rate'),one_time_expenses:num('one-time'),monthly_cost:num('monthly-cost'),contingency_pct:num('contingency')};}
function updateEstimate(){const c=intakePayload();const total=(c.estimated_hours*c.hourly_rate+c.one_time_expenses)*(1+c.contingency_pct/100);const money=n=>Number.isFinite(n)?n.toLocaleString(undefined,{style:'currency',currency:'USD'}):'Invalid';$('estimate-preview').textContent=`Estimated one-time budget: ${money(total)} · Monthly: ${money(c.monthly_cost)} (separate)`;}
const ACTIVE_TASK_STATES = new Set(['queued','processing','failed','blocked']);
function updateBulkSelectionUI(){
 const messages=state.activity?.messages||[];
 const ids=new Set(messages.filter(m=>ACTIVE_TASK_STATES.has(m.status)).map(m=>m.id));
 for(const id of state.selectedTaskIds){if(!ids.has(id))state.selectedTaskIds.delete(id)}
 const selected=state.selectedTaskIds.size;
 $('select-all-queued').disabled=!messages.some(m=>m.status==='queued');
 $('clear-task-selection').disabled=!selected;
 $('cancel-selected-tasks').disabled=!selected;
 $('cancel-selected-tasks').textContent=`Cancel selected (${selected})`;
}
async function refreshActivity(){
 const pid=state.projectSelected;if(!pid)return;
 const data=await api('/api/projects/'+pid+'/activity');if(state.projectSelected!==pid)return;
 state.activity=data;renderArchitecture();$('project-live').hidden=false;
 $('live-title').textContent=data.project.name+' · Live agent activity';
 const counts=data.counts;
 const percentage=counts.total?Math.round(100*counts.done/counts.total):0;
 $('work-progress').style.width=percentage+'%';
const usage=data.llm_usage||{call_count:0,total_tokens:0};
$('token-usage').textContent=`GPT API usage: ${usage.call_count} recorded requests · ${usage.total_tokens.toLocaleString()} reported tokens`;

 $('work-stats').textContent=`${counts.done} agent messages complete · ${counts.queued} queued · ${counts.processing} running · ${counts.failed} failed · ${(counts.cancelled||0)} cancelled · ${counts.total} total. Completion does not imply code approval.`;
 const root=$('activity-sequence');root.replaceChildren();
 if(!data.messages.length){
  root.append(mk('p',{class:'muted'},'No agent work yet. Start the project to queue the Product Manager.'));
  $('activity-detail').textContent='No task selected.';
  $('ask-worker').disabled=true;$('cancel-task').disabled=true;$('retry-task').disabled=true;
  updateBulkSelectionUI();return;
 }
 if(!state.selectedTask||!data.messages.some(x=>x.id===state.selectedTask))state.selectedTask=data.messages[0].id;
 for(const [i,m] of data.messages.entries()){
  const entry=mk('div',{class:'activity-entry'});
  const check=mk('input',{type:'checkbox',class:'task-checkbox','aria-label':`Select ${nice(m.recipient)} task ${i+1} for cancellation`});
  check.disabled=!ACTIVE_TASK_STATES.has(m.status);
  check.checked=state.selectedTaskIds.has(m.id);
  check.addEventListener('change',()=>{if(check.checked)state.selectedTaskIds.add(m.id);else state.selectedTaskIds.delete(m.id);updateBulkSelectionUI()});
  const row=mk('button',{class:'activity-row'+(m.id===state.selectedTask?' selected':''),type:'button'});
  row.append(mk('span',{},`${i+1}. ${nice(m.recipient)}`),mk('small',{class:m.status},m.status==='done'?'Complete':m.status==='processing'?'Running':nice(m.status)));
  row.addEventListener('click',()=>{state.selectedTask=m.id;refreshActivity().catch(e=>alertMessage(e.message))});
  entry.append(check,row);root.append(entry);
 }
 const task=data.messages.find(x=>x.id===state.selectedTask);
 $('activity-role').textContent=nice(task.recipient);
 const errorDetails=task.error?`\n\nError:\n${task.error}`:'';
 const cancelDetails=task.cancel_reason?`\n\nCancellation reason:\n${task.cancel_reason}`:'';
 $('activity-detail').textContent=`Status: ${task.status} · Attempt: ${task.attempts}
Model: ${task.model||'Not selected yet'}

Assignment:
${task.body}

Agent response:
${task.reply||'Not yet available'}${errorDetails}${cancelDetails}`;
 $('ask-worker').disabled=false;
 $('cancel-task').disabled=!ACTIVE_TASK_STATES.has(task.status);
 $('retry-task').disabled=task.status!=='failed';
 updateBulkSelectionUI();
}
async function refreshRagStatus(){
 const project=state.projects.find(p=>p.id===state.projectSelected);
 const panel=$('rag-panel');panel.hidden=!project||!state.activity||state.activity.project?.intake?.product_type!=='rag';
 if(panel.hidden)return;
 const pid=project.id;
 const data=await api(`/api/projects/${pid}/rag/status`);
 if(state.projectSelected!==pid)return;
 const job=data.job;
 const busy=['queued','running'].includes(job?.status);
 const indexText=data.indexed?`Indexed ${data.chunk_count} passages from ${data.receipt?.documents?.length||0} documents.`:'No index has been built.';
 const jobText=job?`Last index job: ${job.status}${job.error?' — '+job.error:''}.`:'';
 $('rag-state').textContent=indexText+(data.stale?' Index is stale—rebuild before asking.':'')+' '+jobText;
 const verified=(data.receipt?.ingestion_checks?.document_checksums==='verified');
 $('rag-details').textContent=verified?'Ingestion integrity verified. Answer accuracy, user access control, and release approval remain untested.':'Readable PDF/TXT/MD/CSV content and OpenAI embeddings are required. Scanned PDFs need OCR (not available).';
 $('rag-build').disabled=busy||state.demo;
 $('rag-ask').disabled=!data.indexed||data.stale||busy||state.demo;
 $('rag-upload').disabled=busy;
}
let projectToDelete=null;
function openDeleteDialog(project){
 projectToDelete={id:project.id,name:project.name};
 $('delete-description').textContent=`Delete ${project.name}? This permanently removes only this project's files, messages, RAG index and meetings.`;
 $('delete-confirm-name').value='';$('delete-project-confirm').disabled=true;
 $('delete-dialog').hidden=false;$('delete-confirm-name').focus();
}
function closeDeleteDialog(){$('delete-dialog').hidden=true;projectToDelete=null;}
async function submitDeleteProject(e){
 e.preventDefault();const project=projectToDelete;
 if(!project||$('delete-confirm-name').value!==project.name)return;
 $('delete-project-confirm').disabled=true;
 try{
  await post(`/api/projects/${project.id}/delete`,{confirmation:project.name});
  if(state.projectSelected===project.id){state.projectSelected=null;state.selectedTask=null;state.activity=null;state.selectedTaskIds.clear();}
  if(state.room?.project_id===project.id){state.room=null;state.selected=null;state.renderedIds='';}
  closeDeleteDialog();await refreshProjects();await refreshMeetings();
  if(state.selected)await refreshRoom();else renderRoom();
  $('rag-answer').textContent='Project deleted. No indexed documents are selected.';
  alertMessage(`Project '${project.name}' permanently deleted.`);
 }catch(err){alertMessage(err.message);$('delete-project-confirm').disabled=false;}
}
async function uploadRagDocuments(){
 const pid=state.projectSelected;const files=[...$('rag-files').files];
 if(!pid||!files.length){alertMessage('Choose a project and documents first.');return;}
 if(files.some(f=>f.size<1||f.size>8*1024*1024)){alertMessage('Each file must be between 1 byte and 8 MiB.');return;}
 $('rag-upload').disabled=true;
 try{
  for(const file of files){await post(`/api/projects/${pid}/artifacts`,{name:file.name,base64:await fileToBase64(file)})}
  $('rag-files').value='';await refreshRagStatus();
  alertMessage(`${files.length} documents uploaded. Rebuild the index to include them.`);
 }catch(err){alertMessage(err.message)}finally{$('rag-upload').disabled=false}
}
async function buildRagIndex(){
 const pid=state.projectSelected;if(!pid)return;
 $('rag-build').disabled=true;
 try{const result=await post(`/api/projects/${pid}/rag/index`,{});
  $('rag-state').textContent=`Indexing requested (${result.job_id}). Document extraction and embeddings are running.`;
  await refreshRagStatus();
 }catch(err){alertMessage(err.message);await refreshRagStatus()}
}
async function askRag(e){
 e.preventDefault();const pid=state.projectSelected;const question=$('rag-question').value.trim();if(!pid||!question)return;
 $('rag-ask').disabled=true;$('rag-answer').textContent='Retrieving passages and requesting an OpenAI answer…';
 try{
  const result=await post(`/api/projects/${pid}/rag/ask`,{question});
  const root=$('rag-answer');root.replaceChildren();root.append(mk('h3',{},'Grounded answer'),mk('p',{class:'rag-generated-answer'},result.answer),mk('h3',{},'Retrieved source references'));
  for(const source of result.sources){const el=mk('div',{class:'rag-source'});
   el.append(mk('b',{},`${source.ref} · ${source.file}${source.page?' · page '+source.page:''}`),mk('p',{},source.preview));root.append(el)}
  root.append(mk('p',{class:'muted'},result.note));
 }catch(err){$('rag-answer').textContent='RAG request failed: '+err.message;alertMessage(err.message)}finally{await refreshRagStatus()}
}
async function refreshEvidence(){
 const pid=state.projectSelected;
 $('handoff-heading').textContent=pid?'Evidence for '+(state.projects.find(p=>p.id===pid)?.name||pid):'Project artifact ledger';
 const filesRoot=$('artifact-list'),handoffRoot=$('handoff-list');filesRoot.replaceChildren();handoffRoot.replaceChildren();
 if(!pid){filesRoot.textContent='Create or select a project first.';handoffRoot.textContent='No project selected.';return}
 const [a,h]=await Promise.all([api(`/api/projects/${pid}/artifacts`),api(`/api/projects/${pid}/handoffs`)]);
 if(state.projectSelected!==pid)return;
 if(!a.artifacts.length)filesRoot.textContent='No documents uploaded yet.';
 for(const file of a.artifacts){const el=mk('div',{class:'evidence-item'});el.append(mk('b',{},file.name),mk('small',{},`${file.size} bytes · SHA-256 ${file.sha256}`));filesRoot.append(el)}
 if(!h.handoffs.length)handoffRoot.textContent='No agent handoffs yet.';
 for(const item of h.handoffs){const el=mk('div',{class:'evidence-item'});el.append(mk('b',{},`${nice(item.sender)} → ${nice(item.recipient)}`),mk('small',{},`${item.status} · integrity ${item.integrity} · tests ${item.manifest.test_status}`));handoffRoot.append(el)}
}
function renderRooms(){const root=$('rooms');root.replaceChildren();$('metric-meetings').textContent=state.meetings.length;if(!state.meetings.length){root.append(mk('p',{class:'muted'},'No meetings yet.'));return;}for(const m of state.meetings){const b=mk('button',{class:'room-item'+(state.selected===m.id?' selected':'')});b.append(mk('div',{class:'room-title'},m.topic),mk('div',{class:'room-info'},m.status.toUpperCase()+' · '+m.participants.length+' agents · '+time(m.starts_at)));b.addEventListener('click',async()=>{state.selected=m.id;state.renderedIds='';try{await refreshRoom()}catch(e){alertMessage(e.message)}});root.append(b);}}
async function refreshMeetings(){state.meetings=(await api('/api/meetings')).meetings;if(!state.selected&&state.meetings.length)state.selected=state.meetings[0].id;renderRooms();}
function renderRoom(){const room=state.room;const live=room?.status==='live';const human=live&&room.floor==='human';$('take-floor').disabled=!live||human;$('yield-floor').disabled=!human;$('dictate').disabled=!live||human;$('end-room').disabled=!live&&room?.status!=='scheduled';$('cancel-room').disabled=!room||['ended','cancelled'].includes(room.status);$('send-question').disabled=!live;
 $('room-title').textContent=room?.topic||'Select a meeting';$('room-subtitle').textContent=room?`Chaired by Product Manager · ${room.participants.length} AI participants · ID ${room.id}`:'Create a room or select an existing one to begin.';
 const badge=$('room-badge');badge.textContent=!room?'Not connected':human?'Human speaking':room.status==='live'?'Live':room.status==='scheduled'?'Scheduled':room.status==='cancelled'?'Cancelled':'Ended';badge.className='status status-'+(!room?'idle':human?'human':room.status);
 const banner=$('floor-banner');banner.className='floor-banner'+(human?' human':'');banner.textContent=!room?'Start a meeting to open the live conversation.':room.status==='scheduled'?'Scheduled for '+new Date(room.starts_at*1000).toLocaleString():room.status==='ended'?'Meeting closed. Transcript saved.':room.status==='cancelled'?'Meeting cancelled. Pending agent turns were stopped and preserved in the transcript.':human?'✋ You have the floor. Agent messages are paused, including answers already in progress.':'● Agents have the floor. A directed question receives priority over the general agenda.';
 $('participant-count').textContent=(room?.participants.length||0)+' agents';const people=$('participants');people.replaceChildren();for(const id of room?.participants||[]){people.append(mk('span',{class:'participant'+(id==='product-manager'?' pm':'')},nice(id)+(id==='product-manager'?' · Chair':'')));}
 fillSelect($('speak-target'),Object.keys(state.roles).map(id=>[id,nice(id)]),['all','Everyone (PM responds)']);
 const transcript=$('transcript');const messages=room?.messages||[];const signature=(room?.id||'none')+':'+messages.map(m=>m.id).join(',');if(state.renderedIds===signature)return;
 const bottom=transcript.scrollHeight-transcript.scrollTop-transcript.clientHeight<70;const previousHeight=transcript.scrollHeight;state.renderedIds=signature;transcript.replaceChildren();
 if(!room){const e=mk('div',{class:'empty'});e.append(mk('div',{class:'empty-icon'},'◉'),mk('b',{},'Bring your AI team together'),mk('p',{},'Any role can join a discussion; the Product Manager moderates and you control the floor.'));transcript.append(e);return;}
 for(const m of messages){const type=m.actor==='human'?'human':m.actor==='system'?'system':'agent';const item=mk('article',{class:'message '+type});item.append(mk('div',{class:'message-avatar'},m.actor==='human'?'YOU':m.actor==='system'?'i':m.actor==='product-manager'?'PM':m.actor.slice(0,2).toUpperCase()));const bubble=mk('div',{class:'bubble'});const by=mk('div',{class:'byline'});by.append(mk('strong',{},m.actor==='human'?'You':nice(m.actor)),mk('span',{},time(m.created_at)));if(m.target!=='all')by.append(mk('em',{},'→ '+nice(m.target)));if(m.kind==='question')by.append(mk('em',{},'Question'));bubble.append(by,mk('p',{},m.body));item.append(bubble);transcript.append(item);}
 if(bottom)transcript.scrollTop=transcript.scrollHeight;else transcript.scrollTop+=Math.max(0,transcript.scrollHeight-previousHeight);
}
async function refreshRoom(){if(!state.selected)return;state.room=await api('/api/meetings/'+state.selected);renderRoom();renderRooms();}
async function performRoom(action,body={}){if(!state.selected)return;try{state.room=await post(`/api/meetings/${state.selected}/${action}`,body);renderRoom();await refreshMeetings();}catch(e){alertMessage(e.message);}}
function openModal(){const roles=$('role-checks');roles.replaceChildren();for(const id of Object.keys(state.roles)){const label=mk('label',{class:'role-check'});const input=mk('input',{type:'checkbox',value:id});if(['rag-engineer','agentic-ai-engineer'].includes(id))input.checked=true;input.addEventListener('change',countChecked);label.append(input,mk('span',{},nice(id)));roles.append(label);}countChecked();$('modal').hidden=false;$('meeting-topic').focus();}
function countChecked(){$('checked-count').textContent=document.querySelectorAll('#role-checks input:checked').length+' selected';}
function closeModal(){$('modal').hidden=true;}
function fileToBase64(file){return new Promise((resolve,reject)=>{
 const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',',2)[1]);
 reader.onerror=()=>reject(Error('Could not read '+file.name));reader.readAsDataURL(file);
})}
let pendingCancellation=null;
let cancelFocus=null;
function openCancelDialog(kind,options){
 cancelFocus=document.activeElement;
 pendingCancellation={kind, roomId:state.selected, projectId:state.projectSelected,
  taskId:state.selectedTask, taskIds:[...state.selectedTaskIds]};
 $('cancel-heading').textContent=options.title;
 $('cancel-description').textContent=options.description;
 $('cancel-reason').value='';
 $('cancel-reason').required=options.reasonRequired;
 $('cancel-reason-label').hidden=!options.reasonRequired;
 $('cancel-reason').hidden=!options.reasonRequired;
 $('cancel-confirm').textContent=kind==='end'?'End meeting':'Confirm cancellation';
 $('cancel-confirm').disabled=false;
 $('cancel-dialog').hidden=false;
 (options.reasonRequired?$('cancel-reason'):$('cancel-confirm')).focus();
}
function closeCancelDialog(){
 $('cancel-dialog').hidden=true;pendingCancellation=null;
 if(cancelFocus?.isConnected)cancelFocus.focus();
}
async function submitCancelDialog(event){
 event.preventDefault();const pending=pendingCancellation;if(!pending)return;
 const reason=$('cancel-reason').value.trim();
 if(pending.kind!=='end'&&reason.length<5){alertMessage('Enter a reason containing at least 5 characters.');$('cancel-reason').focus();return}
 $('cancel-confirm').disabled=true;
 try{
  if(pending.kind==='meeting'||pending.kind==='end'){
   state.room=await post(`/api/meetings/${pending.roomId}/${pending.kind==='end'?'end':'cancel'}`,
    pending.kind==='end'?{}:{reason});
   await refreshMeetings();renderRoom();
  }else if(pending.kind==='task'){
   await post(`/api/tasks/${pending.taskId}/cancel`,{reason});await refreshActivity();
  }else if(pending.kind==='bulk'){
   const result=await post(`/api/projects/${pending.projectId}/cancel-tasks`,{task_ids:pending.taskIds,reason});
   state.selectedTaskIds.clear();await refreshActivity();alertMessage(`${result.cancelled} selected task(s) cancelled.`);
  }
  closeCancelDialog();
 }catch(e){alertMessage(e.message);$('cancel-confirm').disabled=false}
}
async function init(){try{
 const [architecture,health,routeSettings]=await Promise.all([api('/api/architecture'),api('/api/health'),api('/api/settings')]);state.roles=architecture.roles;state.layers=architecture.layers;state.demo=health.offline_demo;$('metric-roles').textContent=Object.keys(state.roles).length;$('demo-label').textContent=health.offline_demo?'OFFLINE DEMO — no GPT calls.':health.api_key_configured?'OpenAI API key configured.':'OPENAI_API_KEY not configured.';renderModelRouting(routeSettings);renderArchitecture();
 await Promise.all([refreshProjects(),refreshMeetings()]);await refreshRoom();
}catch(e){alertMessage('Initialization failed: '+e.message);}}
function bind(){
 $('workflow-menu').addEventListener('click',()=>{document.body.classList.toggle('navigation-open');$('navigation-backdrop').hidden=!document.body.classList.contains('navigation-open')});
 $('navigation-backdrop').addEventListener('click',()=>{document.body.classList.remove('navigation-open');$('navigation-backdrop').hidden=true});
 $('route-configure').addEventListener('click',()=>switchView('settings'));
 $('project-provider-settings').addEventListener('click',()=>switchView('settings'));
 $('project-save-key').addEventListener('click',async()=>{
  const key=$('project-api-key').value.trim();if(!key){alertMessage('Enter an OpenAI API key.');return;}
  try{const result=await post('/api/settings/key',{api_key:key});$('project-api-key').value='';
    $('project-key-status').textContent='OpenAI key saved: '+result.key_storage+'.';alertMessage('Provider key saved on the server.');}
  catch(e){alertMessage(e.message)}
 });
 $('project-clear-key').addEventListener('click',async()=>{
  if(!confirm('Remove the saved OpenAI key and switch to demo mode?'))return;
  try{await post('/api/settings/key/clear',{});state.demo=true;$('project-key-status').textContent='No provider key saved.';alertMessage('Key cleared; offline mode enabled.');}
  catch(e){alertMessage(e.message)}
 });
 $('workflow-project-picker').addEventListener('change',async e=>{state.projectSelected=e.target.value||null;state.selectedTask=null;state.selectedTaskIds.clear();renderProjects();if(state.projectSelected)await refreshActivity();else{state.activity=null;renderArchitecture();}});
 document.querySelectorAll('.nav').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.view)));
 $('new-meeting').addEventListener('click',openModal);for(const id of ['close-modal','cancel-modal','close-backdrop'])$(id).addEventListener('click',closeModal);
 $('refresh-rooms').addEventListener('click',async()=>{try{await refreshMeetings();await refreshRoom()}catch(e){alertMessage(e.message)}});
 $('refresh-projects').addEventListener('click',()=>refreshProjects().catch(e=>alertMessage(e.message)));
 $('rag-refresh').addEventListener('click',()=>refreshRagStatus().catch(e=>alertMessage(e.message)));
 $('rag-build').addEventListener('click',buildRagIndex);
 $('rag-upload').addEventListener('click',uploadRagDocuments);
 $('rag-question-form').addEventListener('submit',askRag);
 $('delete-confirm-name').addEventListener('input',()=>{$('delete-project-confirm').disabled=!projectToDelete||$('delete-confirm-name').value!==projectToDelete.name});
 $('delete-project-form').addEventListener('submit',submitDeleteProject);
 for(const id of ['delete-dialog-x','delete-dismiss','delete-backdrop'])$(id).addEventListener('click',closeDeleteDialog);

 $('project-files').addEventListener('change',()=>{$('pending-docs').textContent=[...$('project-files').files].map(f=>`${f.name} (${f.size} bytes)`).join(' · ')||'No files selected.'});
 $('refresh-activity').addEventListener('click',()=>refreshActivity().catch(e=>alertMessage(e.message)));
 $('refresh-handoffs').addEventListener('click',()=>refreshEvidence().catch(e=>alertMessage(e.message)));
 $('open-handoffs').addEventListener('click',()=>switchView('handoffs'));
 if($('hero-project'))$('hero-project').addEventListener('click',()=>switchView('projects'));
 if($('hero-meetings'))$('hero-meetings').addEventListener('click',()=>switchView('meetings'));
 for(const id of ['estimated-hours','hourly-rate','one-time','monthly-cost','contingency'])$(id).addEventListener('input',updateEstimate);
 updateEstimate();
 $('project-form').addEventListener('submit',async e=>{e.preventDefault();const kickoff=$('project-kickoff').checked;const intake=intakePayload();if(kickoff&&!state.demo&&!intake.provider_consent){alertMessage('Check the provider consent box before sending a brief to GPT.');return;}try{
  const docs=[...$('project-files').files];
  if(docs.some(f=>f.size<1||f.size>8*1024*1024))throw Error('Every project file must be between 1 byte and 8 MiB.');
  // Create first, upload references, then kickoff so the project starts with attached documents.
  const project=await post('/api/projects',{name:$('project-name').value,brief:$('project-brief').value,intake,kickoff:false});
  state.projectSelected=project.id;state.selectedTask=null;state.selectedTaskIds.clear();
  try{
   for(const file of docs){const b64=await fileToBase64(file);await post(`/api/projects/${project.id}/artifacts`,{name:file.name,base64:b64})}
  }catch(uploadError){throw Error(`Project ${project.name} was created, but document upload failed: ${uploadError.message}. Please review the artifact ledger before starting.`)}
  if(kickoff)await post(`/api/projects/${project.id}/kickoff`,{});
  $('project-form').reset();$('pending-docs').textContent='Up to 8 MiB each. Files are stored in the project artifact ledger.';
  updateEstimate();await refreshProjects();await refreshActivity();
  alertMessage('Project created'+(kickoff?' and Product Manager queued.':'.')+' Track work below.');
 }catch(err){alertMessage(err.message)}});
 $('ask-worker').addEventListener('click',()=>{const task=state.activity?.messages?.find(x=>x.id===state.selectedTask);if(!task)return;openModal();$('meeting-topic').value=`Review ${state.activity.project.name} with ${nice(task.recipient)}`;$('meeting-project').value=state.projectSelected;for(const el of document.querySelectorAll('#role-checks input'))el.checked=el.value===task.recipient;countChecked();});
 $('meeting-form').addEventListener('submit',async e=>{e.preventDefault();const participants=[...document.querySelectorAll('#role-checks input:checked')].map(x=>x.value);const topic=$('meeting-topic').value;const starts=$('meeting-time').value;const starts_at=starts?Math.floor(new Date(starts).getTime()/1000):null;try{const room=await post('/api/meetings',{topic,participants,project_id:$('meeting-project').value||null,starts_at});state.selected=room.id;state.renderedIds='';closeModal();$('meeting-form').reset();switchView('meetings');await refreshMeetings();await refreshRoom();}catch(err){alertMessage(err.message)}});
 $('take-floor').addEventListener('click',()=>performRoom('interrupt'));
 $('yield-floor').addEventListener('click',()=>performRoom('yield'));
 $('end-room').addEventListener('click',()=>openCancelDialog('end',{
  title:'End this meeting?',description:'End the room and preserve the transcript. All unfinished turns will be discarded.',reasonRequired:false}));
 $('cancel-room').addEventListener('click',()=>openCancelDialog('meeting',{
  title:'Cancel meeting?',description:`Cancel ${state.room?.topic||'this meeting'}? All queued and in-flight turns will be blocked.`,reasonRequired:true}));
 $('speak-form').addEventListener('submit',async e=>{e.preventDefault();const message=$('speak-message').value.trim();if(!message)return;const target=$('speak-target').value;try{state.room=await post(`/api/meetings/${state.selected}/speak`,{message,target});$('speak-message').value='';renderRoom();state.room=await post(`/api/meetings/${state.selected}/yield`,{});renderRoom();await refreshMeetings();}catch(err){alertMessage(err.message)}});
 $('select-all-queued').addEventListener('click',()=>{
  for(const task of state.activity?.messages||[]){if(task.status==='queued')state.selectedTaskIds.add(task.id)}
  refreshActivity().catch(e=>alertMessage(e.message));
 });
 $('clear-task-selection').addEventListener('click',()=>{state.selectedTaskIds.clear();refreshActivity().catch(e=>alertMessage(e.message))});
 $('cancel-selected-tasks').addEventListener('click',()=>{
  if(!state.selectedTaskIds.size)return;
  openCancelDialog('bulk', {title:`Cancel ${state.selectedTaskIds.size} selected tasks?`,
   description:'Only the checked unfinished tasks in this project will be cancelled. Completed tasks remain unchanged.',reasonRequired:true});
 });
 $('retry-task').addEventListener('click',async()=>{
  const task=state.activity?.messages?.find(x=>x.id===state.selectedTask);
  if(!task||task.status!=='failed')return;
  $('retry-task').disabled=true;
  try{
   await post(`/api/projects/${state.projectSelected}/retry`,{message_id:task.id});
   await refreshActivity();
   alertMessage('Failed task requeued. A live GPT retry may incur another API charge.');
  }catch(e){alertMessage(e.message);await refreshActivity().catch(()=>{});}
 });
 $('cancel-task').addEventListener('click',()=>{
  const task=state.activity?.messages?.find(x=>x.id===state.selectedTask);if(!task)return;
  openCancelDialog('task',{title:`Cancel ${nice(task.recipient)} task?`,
   description:'This task cannot publish any later response or delegate new assignments.',reasonRequired:true});
 });
 $('cancel-dialog-x').addEventListener('click',closeCancelDialog);
 $('cancel-dialog-dismiss').addEventListener('click',closeCancelDialog);
 $('cancel-backdrop').addEventListener('click',closeCancelDialog);
 $('cancel-form').addEventListener('submit',submitCancelDialog);
 document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!$('cancel-dialog').hidden)closeCancelDialog()});
 const Recognition=window.SpeechRecognition||window.webkitSpeechRecognition;if(!Recognition){$('dictate').title='Voice dictation is not supported by this browser.';}else{$('dictate').addEventListener('click',async()=>{if(!state.selected)return;try{await performRoom('interrupt');const recognition=new Recognition();recognition.lang=navigator.language||'en-US';recognition.interimResults=true;recognition.continuous=false;$('dictate').disabled=true;recognition.onresult=e=>{$('speak-message').value=[...e.results].map(r=>r[0].transcript).join(' ')};recognition.onerror=e=>alertMessage('Microphone: '+e.error);recognition.onend=()=>{$('dictate').disabled=false;$('speak-message').focus();};recognition.start();}catch(e){alertMessage(e.message)}});}
}
const MODEL_GROUPS=['architecture','engineering','business','routine'];
let currentConfig=null;
const modelValue=id=>$(id).value.trim();
function buildRoleGrid(){
 const root=$('agent-model-grid');root.replaceChildren();
 for(const [role,mandate] of Object.entries(state.roles)){
  const row=mk('label',{class:'agent-model-row'});
  const title=mk('span',{class:'agent-role-title'});
  title.append(mk('b',{},nice(role)),mk('small',{},mandate));
  const group=mk('span',{class:'agent-role-group'},(currentConfig?.roles?.[role]||'').replace(/^$/,'Not configured'));
  const input=mk('input',{id:'worker-model-'+role,'data-role':role,list:'available-models',placeholder:'Group default'});
  input.value=currentConfig?.role_models?.[role]||'';
  row.append(title,group,input);root.append(row);
 }
}
async function refreshSettings(){
 const cfg=await api('/api/settings');currentConfig=cfg;
 $('settings-mode').value=cfg.mode||'demo';
 for(const group of MODEL_GROUPS){$('route-'+group).value=cfg.models?.[group]||'';}
 $('route-meeting').value=cfg.meeting_model||'';
 $('key-status').textContent=cfg.api_key_configured?'OpenAI key configured — stored '+cfg.key_storage+'.':'No API key configured. Demo mode only.';
 buildRoleGrid();renderModelRouting(cfg);
}
function settingsPayload(){
 const role_models={};
 document.querySelectorAll('#agent-model-grid input[data-role]').forEach(input=>{if(input.value.trim())role_models[input.dataset.role]=input.value.trim()});
 const models={};for(const group of MODEL_GROUPS)models[group]=modelValue('route-'+group);
 return {models,role_models,meeting_model:modelValue('route-meeting'),mode:$('settings-mode').value};
}
async function saveSettings(){
 try{const cfg=await post('/api/settings',settingsPayload());currentConfig=cfg;renderModelRouting(cfg);
  state.demo=cfg.mode==='demo';$('demo-label').textContent=state.demo?'OFFLINE DEMO — no GPT calls.':'LIVE OPENAI — API charges may apply.';
  alertMessage('Model routing and run mode saved. GPT usage is tracked without local token caps.');
 }catch(e){alertMessage(e.message)}
}
function bindSettings(){
 $('save-api-key').addEventListener('click',async()=>{
  const key=$('settings-api-key').value.trim();if(!key){alertMessage('Enter an OpenAI API key.');return;}
  try{const result=await post('/api/settings/key',{api_key:key});$('settings-api-key').value='';$('key-status').textContent='OpenAI key configured — stored '+result.key_storage+'.';alertMessage('API key saved on the server.');}catch(e){alertMessage(e.message)}
 });
 $('clear-api-key').addEventListener('click',async()=>{if(!confirm('Clear the saved OpenAI API key and return to offline demo?'))return;
  try{await post('/api/settings/key/clear');state.demo=true;await refreshSettings();alertMessage('Key cleared. Demo mode enabled.');}catch(e){alertMessage(e.message)}
 });
 $('check-models').addEventListener('click',async()=>{
  $('model-lookup-status').textContent='Checking OpenAI model access…';
  try{const data=await api('/api/providers/openai/models');const list=$('available-models');list.replaceChildren();
   for(const id of data.models)list.append(mk('option',{value:id}));
   $('model-lookup-status').textContent=data.models.length+' models listed by OpenAI. Choose supported IDs; not every model supports every structured output feature.';
  }catch(e){$('model-lookup-status').textContent=e.message;alertMessage(e.message)}
 });
 $('save-model-settings').addEventListener('click',saveSettings);
 $('save-role-settings').addEventListener('click',saveSettings);
}

bindSettings();
bind();init();setInterval(async()=>{try{await refreshMeetings();await refreshRoom();if(['projects','architecture'].includes(state.view)&&state.projectSelected){await refreshActivity();if(state.view==='projects')await refreshRagStatus();}}catch(e){/* transient connection errors handled on user actions */}},1500);