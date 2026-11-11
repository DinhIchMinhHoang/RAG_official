import { Hl } from "./Hl.jsx";
import { useI18n } from "../lib/i18n.jsx";

export function EqaView({ex,cand,setCand}){
  const {t}=useI18n();
  const c=ex&&ex.candidates[cand];
  if(!c)return <div className="empty">{t("ins.none")}</div>;
  return <div className="view">
    <div className="sub">{ex.q}</div>
    <div className="score"><span>{t("ins.score")}</span><div className="bar" role="meter" aria-valuemin="0" aria-valuemax="1" aria-valuenow={c.score}><i style={{width:c.score*100+"%"}}/></div><b>{Math.round(c.score*100)}%</b></div>
    <h3>{t("ins.context")}</h3>
    <div className="doc"><Hl text={ex.ctx} start={c.start} end={c.end}/></div>
    <h3>{t("ins.cands")}</h3>
    <div className="cands">{ex.candidates.map((x,i)=><button key={i} className={"cand"+(i===cand?" on":"")} aria-pressed={i===cand} onClick={()=>setCand(i)}><span>{x.text}</span><b>{Math.round(x.score*100)}%</b></button>)}</div>
  </div>;
}
