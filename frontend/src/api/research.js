import {api} from './client.js';
export const sourcesApi={list:()=>api('/sources'),create:data=>api('/sources',{method:'POST',body:JSON.stringify(data)}),update:(id,data)=>api('/sources/'+id,{method:'PUT',body:JSON.stringify(data)}),records:id=>api('/sources/'+id+'/records'),add:(id,records)=>api('/sources/'+id+'/records',{method:'POST',body:JSON.stringify({records})})};
export const runsApi={list:()=>api('/runs'),get:id=>api('/runs/'+id),create:data=>api('/runs',{method:'POST',body:JSON.stringify(data)})};
export const notesApi={list:()=>api('/notes'),create:data=>api('/notes',{method:'POST',body:JSON.stringify(data)}),approve:id=>api('/notes/'+id+'/approve',{method:'POST'})};
export const looApi={list:runId=>api('/loo-jobs?run_id='+runId),get:id=>api('/loo-jobs/'+id),create:(runId,batchSize)=>api('/loo-jobs',{method:'POST',body:JSON.stringify({run_id:runId,batch_size:batchSize||null})})};
