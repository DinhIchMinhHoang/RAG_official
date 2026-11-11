import { useI18n } from "../lib/i18n.jsx";

export function CorpusView({srcs,viewSrc,setViewSrc}){
  const {t}=useI18n();
  const s=srcs.find(x=>x.id===viewSrc)||srcs[0];
  if(!s)return <div className="empty">{t("ins.corpusNone")}</div>;
  return <div className="view">
    <div className="cands">{srcs.map(x=><button key={x.id} className={"cand"+(x.id===s.id?" on":"")} aria-pressed={x.id===s.id} onClick={()=>setViewSrc(x.id)}><span>{x.name}</span><b>{x.type}</b></button>)}</div>
    {s.status!=="ready"?<div className="empty">{t("src.indexing")}</div>:s.pages.map((p,i)=><div key={i}>
      <h3>{t("ins.page",{n:i+1})}</h3>
      <div className="doc">{p}</div>
    </div>)}
  </div>;
}
