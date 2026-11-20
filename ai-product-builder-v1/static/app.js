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
 for(const v of ['meetings','architecture','projects','handoffs'])$('view-'+v).hidden=v!==view;
 document.querySelectorAll('.nav').forEach(n=>n.classList.toggle('active',n.dataset.view===view));
 $('view-name').textContent={meetings:'Meeting studio',architecture:'Organization',projects:'Projects',handoffs:'Handoffs & artifacts'}[view];
 if(view==='handoffs')refreshEvidence().catch(e=>alertMessage(e.message));
}

function fillSelect(select,entries,first){const old=select.value;select.replaceChildren();if(first)select.append(mk('option',{value:first[0]},first[1]));entries.forEach(([value,label])=>select.append(mk('option',{value},label)));if([...select.options].some(o=>o.value===old))select.value=old;}
function renderArchitecture(){const root=$('architecture-layers');root.replaceChildren();for(const layer of state.layers){const panel=mk('div',{class:'layer'});panel.append(mk('div',{class:'eyebrow'},layer.id),mk('h3',{},layer.name));const chips=mk('div',{class:'role-chips'});for(const id of layer.roles)chips.append(mk('span',{class:'role-chip'},nice(id)));panel.append(chips);root.append(panel);}}
function renderProjects(){const root=$('project-list');root.replaceChildren();if(!state.projects.length){root.append(mk('p',{class:'muted'},'No projects yet.'));return;}for(const p of state.projects){const item=mk('button',{class:'project-item'+(state.projectSelected===p.id?' selected':''),type:'button'});item.append(mk('b',{},p.name),mk('small',{},p.status+' · '+p.id));item.addEventListener('click',()=>{state.projectSelected=p.id;state.selectedTask=null;state.selectedTaskIds.clear();renderProjects();refreshActivity().catch(e=>alertMessage(e.message));});root.append(item);}fillSelect($('meeting-project'),state.projects.map(p=>[p.id,p.name]),['','Standalone meeting']);}
async function refreshProjects(){state.projects=(await api('/api/projects')).projects;if(!state.projectSelected&&state.projects.length)state.projectSelected=state.projects[0].id;renderProjects();if(state.projectSelected)await refreshActivity();}
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
 state.activity=data;$('project-live').hidden=false;
 $('live-title').textContent=data.project.name+' · Live agent activity';
 const counts=data.counts;
 const percentage=counts.total?Math.round(100*counts.done/counts.total):0;
 $('work-progress').style.width=percentage+'%';
 $('work-stats').textContent=`${counts.done} agent messages complete · ${counts.queued} queued · ${counts.processing} running · ${counts.failed} failed · ${(counts.cancelled||0)} cancelled · ${counts.total} total. Completion does not imply code approval.`;
 const root=$('activity-sequence');root.replaceChildren();
 if(!data.messages.length){
  root.append(mk('p',{class:'muted'},'No agent work yet. Start the project to queue the Product Manager.'));
  $('activity-detail').textContent='No task selected.';
  $('ask-worker').disabled=true;$('cancel-task').disabled=true;
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
 updateBulkSelectionUI();
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
 const [architecture,health]=await Promise.all([api('/api/architecture'),api('/api/health')]);state.roles=architecture.roles;state.layers=architecture.layers;state.demo=health.offline_demo;$('metric-roles').textContent=Object.keys(state.roles).length;$('demo-label').textContent=health.offline_demo?'OFFLINE DEMO — no GPT calls.':health.api_key_configured?'OpenAI API key configured.':'OPENAI_API_KEY not configured.';renderArchitecture();
 await Promise.all([refreshProjects(),refreshMeetings()]);await refreshRoom();
}catch(e){alertMessage('Initialization failed: '+e.message);}}
function bind(){document.querySelectorAll('.nav').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.view)));
 $('new-meeting').addEventListener('click',openModal);for(const id of ['close-modal','cancel-modal','close-backdrop'])$(id).addEventListener('click',closeModal);
 $('refresh-rooms').addEventListener('click',async()=>{try{await refreshMeetings();await refreshRoom()}catch(e){alertMessage(e.message)}});
 $('refresh-projects').addEventListener('click',()=>refreshProjects().catch(e=>alertMessage(e.message)));
 $('project-files').addEventListener('change',()=>{$('pending-docs').textContent=[...$('project-files').files].map(f=>`${f.name} (${f.size} bytes)`).join(' · ')||'No files selected.'});
 $('refresh-activity').addEventListener('click',()=>refreshActivity().catch(e=>alertMessage(e.message)));
 $('refresh-handoffs').addEventListener('click',()=>refreshEvidence().catch(e=>alertMessage(e.message)));
 $('open-handoffs').addEventListener('click',()=>switchView('handoffs'));
 $('hero-project').addEventListener('click',()=>switchView('projects'));
 $('hero-meetings').addEventListener('click',()=>switchView('meetings'));
 for(const id of ['estimated-hours','hourly-rate','one-time','monthly-cost','contingency'])$(id).addEventListener('input',updateEstimate);
 updateEstimate();
 $('project-form').addEventListener('submit',async e=>{e.preventDefault();const kickoff=$('project-kickoff').checked;const intake=intakePayload();if(kickoff&&!state.demo&&!intake.provider_consent){alertMessage('Check the provider consent box before sending a brief to GPT.');return;}try{
  const docs=[...$('project-files').files];
  if(docs.some(f=>f.size<1||f.size>1_000_000))throw Error('Every project file must be between 1 byte and 1 MB.');
  // Create first, upload references, then kickoff so the project starts with attached documents.
  const project=await post('/api/projects',{name:$('project-name').value,brief:$('project-brief').value,intake,kickoff:false});
  state.projectSelected=project.id;state.selectedTask=null;state.selectedTaskIds.clear();
  try{
   for(const file of docs){const b64=await fileToBase64(file);await post(`/api/projects/${project.id}/artifacts`,{name:file.name,base64:b64})}
  }catch(uploadError){throw Error(`Project ${project.name} was created, but document upload failed: ${uploadError.message}. Please review the artifact ledger before starting.`)}
  if(kickoff)await post(`/api/projects/${project.id}/kickoff`,{});
  $('project-form').reset();$('pending-docs').textContent='Up to 1 MB each. Files are stored in the project artifact ledger.';
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
bind();init();setInterval(async()=>{try{await refreshMeetings();await refreshRoom();if(state.view==='projects'&&state.projectSelected)await refreshActivity();}catch(e){/* transient connection errors handled on user actions */}},1500);
