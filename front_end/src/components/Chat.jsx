import { Icon } from "./Icon.jsx";
import { useI18n } from "../lib/i18n.jsx";

const EXAMPLES={
  eqa:["Who introduced BERT?","What does the retriever find?","When was BERT introduced?"],
  rag:["What is retrieval-augmented generation?","How does the generator use passages?","Why ground answers in a corpus?"]
};

export function Chat({exs,cur,mode,canAsk,onSelect,onCite,copied,copy,end,onExample,onRetry,busy,stick}){
  const {t}=useI18n();
  // rag answers carry inline "[n]" markers: render them as clickable citation chips
  const body=m=>m.mode==="rag"?m.answer.split(/(\[\d+\])/).map((p,i)=>{
    const n=/^\[(\d+)\]$/.exec(p);
    if(!n)return p;
    // while streaming the citation data does not exist yet: plain chip, clickable only once done
    return m.phase==="done"?<button key={i} className="cite" aria-label={t("chat.cite",{n:n[1]})} onClick={e=>{e.stopPropagation();onCite(m,+n[1]);}}>{p}</button>:<span key={i} className="cite">{p}</span>;
  }):m.answer;
  return <div className="chat" onScroll={e=>{const c=e.currentTarget;stick.current=c.scrollHeight-c.scrollTop-c.clientHeight<80;}}>
    {!exs.length&&<div className="start">
      <div className="empty">{t("chat.start."+mode)}</div>
      <div className="chips">{EXAMPLES[mode].map(p=><button key={p} className="chip" disabled={busy||!canAsk} onClick={()=>onExample(p)}>{p}</button>)}</div>
    </div>}
    {exs.map(m=><div key={m.id} className={"ex"+(cur&&cur.id===m.id?" sel":"")} onClick={()=>onSelect(m)}>
      <div className="bubble user">{m.q}</div>
      {m.phase==="error"?<div className="ai err" role="alert">
        <div>{t("chat.error")}</div>
        <button className="retry" disabled={busy} onClick={()=>onRetry(m)}><Icon n="retry"/>{t("chat.retry")}</button>
      </div>:<div className="ai">
        {m.phase==="typing"?<div className="dots" role="status" aria-label={t("chat.thinking")}><i/><i/><i/></div>:<>
          <div aria-busy={m.phase==="streaming"}>{m.answer||m.phase!=="done"?<>{body(m)}{m.phase==="streaming"&&<span className="cursor" aria-hidden="true"/>}</>:t("chat.noans")}</div>
          {m.phase!=="streaming"&&<div className="meta">
            {m.phase==="stopped"?<><span>{t("chat.stopped")}</span><button className="retry" disabled={busy} onClick={e=>{e.stopPropagation();onRetry(m);}}><Icon n="retry"/>{t("chat.retry")}</button></>:<>
              {m.mode==="eqa"&&m.answer&&<span>{Math.round(m.score*100)}%</span>}
              <span>{m.ms} ms</span>
            </>}
            {m.answer&&<button className="iconbtn copy" aria-label={t("chat.copy")} onClick={e=>{e.stopPropagation();copy(m);}}><Icon n={copied===m.id?"check":"copy"}/></button>}
          </div>}
        </>}
      </div>}
    </div>)}
    <div ref={end}/>
  </div>;
}
