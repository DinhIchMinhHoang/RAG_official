import { useI18n } from "../lib/i18n.jsx";

export function ContextBox({ctx,setCtx}){
  const {t}=useI18n();
  return <div className="ctxbox">
    <div className="shead"><span className="lb">{t("ctx.title")}</span>
      <button className="linkbtn" disabled={!ctx} onClick={()=>setCtx("")}>{t("ctx.clear")}</button></div>
    <textarea className="field ctxarea" aria-label={t("ctx.title")} placeholder={t("ctx.ph")} value={ctx} onChange={e=>setCtx(e.target.value)}/>
    <div className="sub">{t("ctx.count",{n:ctx.length})}</div>
  </div>;
}
