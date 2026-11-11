import { Hl } from "./Hl.jsx";
import { useI18n } from "../lib/i18n.jsx";

export function SourceView({ex,cite,setCite,srcs}){
  const {t}=useI18n();
  const cs=ex?ex.citations:[],c=cs.find(x=>x.id===cite)||cs[0];
  if(!c)return <div className="empty">{t("ins.srcNone")}</div>;
  const s=srcs.find(x=>x.id===c.srcId),page=s&&s.pages[c.page-1];
  const at=page?page.indexOf(c.quote):-1;
  return <div className="view">
    <div className="cands row-chips">{cs.map(x=><button key={x.id} className={"cite"+(x.id===c.id?" on":"")} aria-pressed={x.id===c.id} onClick={()=>setCite(x.id)}>[{x.id}]</button>)}</div>
    <div className="score"><span>{t("ins.score")}</span><div className="bar"><i style={{width:c.score*100+"%"}}/></div><b>{Math.round(c.score*100)}%</b></div>
    {s?<>
      <h3>{s.name} · {t("ins.page",{n:c.page})}</h3>
      <div className="doc"><Hl text={page} start={at} end={at<0?-1:at+c.quote.length}/></div>
    </>:<div className="empty">{t("ins.missing")}</div>}
  </div>;
}
