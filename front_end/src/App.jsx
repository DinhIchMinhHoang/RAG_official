import { useState, useEffect, useRef } from "react";
import { streamEqa, streamRag, indexSource, extOf } from "./lib/api.js";
import { SAMPLE_CTX, SAMPLE_SRC } from "./data/sample.js";
import { useI18n } from "./lib/i18n.jsx";
import { useTheme } from "./lib/useTheme.js";
import { useHistory } from "./lib/useHistory.js";
import { Notch } from "./components/Notch.jsx";
import { Sidebar } from "./components/Sidebar.jsx";
import { Chat } from "./components/Chat.jsx";
import { Composer, MAX } from "./components/Composer.jsx";
import { Inspector } from "./components/Inspector.jsx";
import { Resizer, clampW } from "./components/Resizer.jsx";
import { Icon } from "./components/Icon.jsx";

let uid=100;
const MODE_KEY="lumen.mode";
const initMode=()=>{try{return localStorage.getItem(MODE_KEY)==="rag"?"rag":"eqa";}catch{return "eqa";}};
const initW=(k,d)=>{try{const v=+localStorage.getItem(k);return v?clampW(v):d;}catch{return d;}};
const usable=s=>s.status==="ready"&&s.checked;

export default function App(){
  const {t}=useI18n();
  const [theme,setTheme]=useTheme();
  const hist=useHistory();
  const w=window.innerWidth;
  const [mode,setMode]=useState(initMode),[tab,setTab]=useState("history"),[q,setQ]=useState("");
  const [left,setLeft]=useState(w>=800),[right,setRight]=useState(w>=1100);
  const [wl,setWl]=useState(()=>initW("lumen.wl",292)),[wr,setWr]=useState(()=>initW("lumen.wr",372));
  useEffect(()=>{try{localStorage.setItem("lumen.wl",wl);localStorage.setItem("lumen.wr",wr);}catch{}},[wl,wr]);
  const [sid,setSid]=useState({}); // current history session id per mode (undefined = new chat)
  const [exs,setExs]=useState([]); // messages of both modes: {id, mode, q, phase, ...askEqa/askRag result}
  const [ctx,setCtx]=useState(SAMPLE_CTX),[srcs,setSrcs]=useState([SAMPLE_SRC]);
  const [sel,setSel]=useState(null),[cand,setCand]=useState(0),[cite,setCite]=useState(1),[rtab,setRtab]=useState("source"),[viewSrc,setViewSrc]=useState(null);
  const [text,setText]=useState(""),[copied,setCopied]=useState(null),[busy,setBusy]=useState(false);
  const [online,setOnline]=useState(navigator.onLine);
  const [st,setSt]=useState({topK:3,thr:0.3,reduce:false});
  const up=(k,v)=>setSt(s=>({...s,[k]:v}));
  const end=useRef(null),ac=useRef(null),stick=useRef(true); // stick: false once the user scrolls up
  useEffect(()=>{try{localStorage.setItem(MODE_KEY,mode);}catch{}},[mode]);
  useEffect(()=>{document.body.classList.toggle("reduce",st.reduce);},[st.reduce]);
  useEffect(()=>{if(busy&&!stick.current)return;end.current&&end.current.scrollIntoView({behavior:st.reduce?"auto":"smooth",block:"end"});},[exs,mode]);
  useEffect(()=>{ // pointer-following sheen on every glass panel
    const mv=e=>document.querySelectorAll(".glass").forEach(el=>{const r=el.getBoundingClientRect();el.style.setProperty("--mx",(e.clientX-r.left)+"px");el.style.setProperty("--my",(e.clientY-r.top)+"px");});
    window.addEventListener("mousemove",mv);return()=>window.removeEventListener("mousemove",mv);
  },[]);
  useEffect(()=>{ // collapse/expand sidebars only when a breakpoint is crossed
    let pw=window.innerWidth;
    const rs=()=>{const nw=window.innerWidth;
      if((pw>=1100)!==(nw>=1100))setRight(nw>=1100);
      if((pw>=800)!==(nw>=800))setLeft(nw>=800);
      pw=nw;};
    window.addEventListener("resize",rs);return()=>window.removeEventListener("resize",rs);
  },[]);
  useEffect(()=>{
    const on=()=>setOnline(true),off=()=>setOnline(false);
    window.addEventListener("online",on);window.addEventListener("offline",off);
    return()=>{window.removeEventListener("online",on);window.removeEventListener("offline",off);};
  },[]);
  useEffect(()=>{ // Ctrl/Cmd+Shift+L toggles mode
    const k=e=>{if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key.toLowerCase()==="l"){e.preventDefault();setMode(m=>m==="eqa"?"rag":"eqa");}};
    window.addEventListener("keydown",k);return()=>window.removeEventListener("keydown",k);
  },[]);

  const can=m=>m==="eqa"?!!ctx.trim():srcs.some(usable);
  const stop=()=>ac.current&&ac.current.abort();
  const ask=async(id,question,m,sessionId)=>{
    const ctl=ac.current=new AbortController();
    setBusy(true);stick.current=true;
    const patch=p=>setExs(x=>x.map(e=>e.id===id?{...e,...p}:e));
    let acc="";
    try{
      const opts={topK:st.topK,threshold:st.thr};
      const gen=m==="eqa"?streamEqa({question,context:ctx,opts},{signal:ctl.signal}):streamRag({question,sources:srcs.filter(usable),opts},{signal:ctl.signal});
      for await(const ev of gen){
        if(ev.type==="token"){acc+=ev.text;patch({answer:acc,phase:"streaming"});} // shortcut: one state update per chunk, no rAF batching
        else{
          const done={id,mode:m,q:question,...ev.result,ctx:m==="eqa"?ctx:undefined,phase:"done"};
          patch(done);hist.add(sessionId,m,done,done.ctx);
        }
      }
    }catch{
      if(!ctl.signal.aborted)patch({phase:"error"});
      else if(!ctl.drop){ // stopped: keep the text so far, store it as a final message
        const part={id,mode:m,q:question,answer:acc,phase:"stopped",ctx:m==="eqa"?ctx:undefined};
        patch(part);hist.add(sessionId,m,part,part.ctx);
      }
    }finally{setBusy(false);}
  };
  const send=async(override,m=mode)=>{
    const question=(typeof override==="string"?override:text).trim();if(!question||busy||question.length>MAX||!can(m))return;
    const id="e"+Date.now().toString(36)+(uid++),sessionId=sid[m]||"h"+id; // unique across reloads: stored msgs keep their ids
    setSid(s=>({...s,[m]:sessionId}));
    setText("");setExs(x=>[...x,{id,mode:m,q:question,phase:"typing"}]);setSel(null);setCand(0);setCite(1);
    await ask(id,question,m,sessionId);
  };
  const retry=m=>{if(!busy){setExs(x=>x.map(e=>e.id===m.id?{...e,phase:"typing",answer:""}:e));ask(m.id,m.q,m.mode,sid[m.mode]);}};
  const drop=()=>{if(ac.current){ac.current.drop=true;ac.current.abort();}}; // chat replaced: abort without storing
  const reset=()=>{drop();setExs(x=>x.filter(e=>e.mode!==mode));setSid(s=>({...s,[mode]:undefined}));setSel(null);};
  const load=h=>{ // replace that mode's chat with the stored session; later messages continue it
    drop();setMode(h.mode);setExs(x=>[...x.filter(e=>e.mode!==h.mode),...(h.msgs||[]).map(m=>({...m,mode:h.mode}))]);
    setSid(s=>({...s,[h.mode]:h.id}));if(h.ctx!=null)setCtx(h.ctx);setSel(null);setCand(0);setCite(1);if(window.innerWidth<800)setLeft(false);};
  const select=m=>{setSel(m.id);setCand(0);setCite(1);};
  const openCite=(m,n)=>{setSel(m.id);setCite(n);setRtab("source");setRight(true);};
  const copy=m=>{navigator.clipboard&&navigator.clipboard.writeText(m.answer).then(()=>{setCopied(m.id);setTimeout(()=>setCopied(null),1200);}).catch(()=>{});};

  const corpus={
    srcs,
    add:files=>[...files].forEach(f=>{
      const id="s"+(uid++);
      setSrcs(s=>[...s,{id,name:f.name,type:extOf(f.name),status:"indexing",checked:true,pages:[]}]);
      indexSource(f).then(({pages})=>setSrcs(s=>s.map(x=>x.id===id?{...x,pages,status:"ready"}:x))).catch(()=>setSrcs(s=>s.filter(x=>x.id!==id)));
    }),
    toggle:id=>setSrcs(s=>s.map(x=>x.id===id?{...x,checked:!x.checked}:x)),
    remove:id=>setSrcs(s=>s.filter(x=>x.id!==id)),
    open:id=>{setViewSrc(id);setRtab("corpus");setRight(true);}
  };
  const list=exs.filter(e=>e.mode===mode);
  const cur=list.find(e=>e.id===sel)||list[list.length-1];

  return <>
    {!online&&<div className="banner" role="status">{t("net.offline")}</div>}
    <Notch mode={mode} setMode={setMode} status={!online?"error":busy?"thinking":"ready"}/>
    <button className="iconbtn tbtn l" aria-label={t("side.toggle")} aria-expanded={left} onClick={()=>setLeft(v=>!v)}><Icon n="panel"/></button>
    <button className="iconbtn tbtn r" aria-label={t("insp.toggle")} aria-expanded={right} onClick={()=>setRight(v=>!v)}><Icon n="panelR"/></button>
    {left&&<div className="scrim l" onClick={()=>setLeft(false)}/>}
    {right&&<div className="scrim r" onClick={()=>setRight(false)}/>}

    <div className="app" style={{"--wl":wl+"px","--wr":wr+"px"}}>
      <Sidebar w={wl} setW={setWl} left={left} mode={mode} ctx={ctx} setCtx={setCtx} corpus={corpus} tab={tab} setTab={setTab} q={q} setQ={setQ} onNew={reset} onLoad={load} st={st} up={up} hist={hist} theme={theme} setTheme={setTheme}/>

      <main className="mid">
        <Chat exs={list} cur={cur} mode={mode} canAsk={can(mode)} onSelect={select} onCite={openCite} copied={copied} copy={copy} end={end} onExample={send} onRetry={retry} busy={busy} stick={stick}/>
        <Composer mode={mode} canAsk={can(mode)} text={text} setText={setText} busy={busy} send={send} stop={stop}/>
      </main>

      <aside className={"side right glass"+(right?"":" closed")} aria-hidden={!right} inert={!right}><div className="in">
        <Inspector mode={mode} ex={cur&&cur.phase==="done"?cur:null} cand={cand} setCand={setCand} cite={cite} setCite={setCite} tab={rtab} setTab={setRtab} srcs={srcs} viewSrc={viewSrc} setViewSrc={setViewSrc}/>
      </div><Resizer side="right" value={wr} onChange={setWr} reset={()=>setWr(372)}/></aside>
    </div>
    {hist.undo&&<div className="toast glass" role="status"><span>{t("hist.deleted")}</span><button onClick={hist.restore}>{t("hist.undo")}</button></div>}
  </>;
}
