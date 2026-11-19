import {looApi} from '../api/research.js';
import {session} from '../stores/session.js';
import {status} from './common/status.js';
import {esc} from '../utils/format.js';
import {bind,toast} from '../hooks/actions.js';
import {download} from '../api/client.js';

// 逐站剔除影响面板：只在已完成（已冻结）研究上出现；排名仅在后端校验齐全后发布
export async function looPanel(runId){
  let jobs=[];
  try{jobs=await looApi.list(runId)}catch(e){return ''}
  const job=jobs[0];
  const head='<section class="loo-panel"><h2>复访前逐站剔除影响 <small>剔除一座微站，剩余样本重新标准化，邻域参数照旧</small></h2>';
  if(!job){
    return head+(session.writable
      ?'<form id="loo-form" class="loo-form"><label>每批试算站数（1–5000，默认256）<input name="batch_size" type="number" min="1" max="5000" placeholder="256"></label><p class="fine">额外内存上限 128MB · 不存全站距离矩阵 · 不逐站重跑聚类，基线邻域图仅构建一次；中断后只沿用完好批次，坏批回撤重试不重复计入。</p><button type="submit">发起逐站剔除试算 →</button></form>'
      :'<p class="empty">还没有逐站剔除试算。由分析师或负责人发起。</p>')+'</section>';
  }
  let body='<div class="run-title"><h3>试算任务 #'+esc(String(job.id))+'</h3>'+status(job.state)+'<small>'+esc(String(job.finished))+' / '+esc(String(job.total))+' 站 · 每批 '+esc(String(job.batch_size))+'</small>'+(job.state==='completed'?'<button id="loo-export" class="secondary">下载排名与退化清单 ZIP ↗</button>':'')+'</div>';
  if(job.state==='failed'){
    body+='<div class="failure"><p>'+esc(job.error||'试算失败')+'</p><p class="fine">失败不影响已冻结研究；修正后可重新发起。</p></div>';
  }else if(!job.result){
    const pct=Math.round((job.progress||0)*100);
    body+='<div class="pending"><span class="spinner"></span><p>分批试算进行中，进度 '+pct+'%（'+esc(String(job.finished))+'/'+esc(String(job.total))+'）。页面自动刷新；中断恢复时完好批次直接沿用。</p><div class="progress"><i style="width:'+pct+'%"></i></div></div>';
  }else{
    const r=job.result;
    body+='<p class="fine">'+esc(r.method)+'</p>';
    body+='<div class="loo-metrics"><span>参评站数 <b>'+esc(String(r.evaluated_stations))+'</b></span><span>尺度退化单列 <b>'+esc(String(r.degenerate_stations))+'</b></span><span>基线噪声站 <b>'+esc(String(r.baseline_noise_count))+'</b></span><span>watch 邻域边 <b>'+esc(String(r.watch_edges))+'</b></span><span>引擎内存峰值 <b>'+(r.memory?Math.round(r.memory.peak_bytes/1024/1024):'-')+'MB / 128MB</b></span></div>';
    const top=r.ranking.slice(0,15);
    body+='<h3>噪声身份翻转排名 <small>按其他站翻转数降序，预览前15名；ZIP 含完整排名</small></h3>';
    body+='<div class="table-wrap"><table><thead><tr><th>排名</th><th>剔除微站</th><th>其他站噪声翻转数</th><th>试算前噪声数</th><th>试算后噪声数</th></tr></thead><tbody>'
      +top.map((it,i)=>'<tr><td>'+(i+1)+'</td><td>'+esc(it.reference)+'</td><td><b>'+esc(String(it.noise_flips))+'</b></td><td>'+esc(String(it.noise_before))+'</td><td>'+esc(String(it.noise_after))+'</td></tr>').join('')
      +'</tbody></table></div>';
    if(r.degenerate_trials.length){
      body+='<h3>尺度退化试算（不参与排名）</h3><div class="table-wrap"><table><thead><tr><th>剔除微站</th><th>单列原因</th></tr></thead><tbody>'
        +r.degenerate_trials.map(it=>'<tr><td>'+esc(it.reference)+'</td><td>'+esc(it.reason)+'</td></tr>').join('')+'</tbody></table></div>';
    }else{
      body+='<p class="fine">本次无尺度退化试算，全部微站均参与排名。</p>';
    }
  }
  return head+body+'</section>';
}

export function bindLoo(runId,refresh){
  bind('#loo-form','submit',async e=>{
    e.preventDefault();
    const v=e.target.batch_size.value;
    try{
      await looApi.create(runId,v?Number(v):null);
      toast('逐站剔除试算已入队');
      await refresh();
    }catch(err){toast(err.message,true)}
  });
  bind('#loo-export','click',async()=>{
    try{await download('/loo-jobs/'+(await looApi.list(runId))[0].id+'/export','loo-ranking-'+runId+'.zip')}
    catch(e){toast(e.message,true)}
  });
}
