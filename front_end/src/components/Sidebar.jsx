import { Seg } from "./Seg.jsx";
import { History } from "./History.jsx";
import { Settings } from "./Settings.jsx";
import { ContextBox } from "./ContextBox.jsx";
import { Sources } from "./Sources.jsx";
import { Resizer } from "./Resizer.jsx";
import { useI18n } from "../lib/i18n.jsx";

export function Sidebar({left,mode,ctx,setCtx,corpus,tab,setTab,q,setQ,onNew,onLoad,st,up,hist,theme,setTheme,w,setW}){
  const {t}=useI18n();
  return <aside className={"side left glass"+(left?"":" closed")} aria-hidden={!left} inert={!left}><div className="in">
    {mode==="eqa"?<ContextBox ctx={ctx} setCtx={setCtx}/>:<Sources corpus={corpus}/>}
    <Seg className="seg" aria-label={t("tab.history")+" / "+t("tab.settings")}value={tab} onValueChange={setTab} items={[{value:"history",label:t("tab.history")},{value:"settings",label:t("tab.settings")}]}/>
    <div className="pane" key={tab}>
      {tab==="history"?<History q={q} setQ={setQ} onNew={onNew} onLoad={onLoad} hist={hist} mode={mode}/>:<Settings st={st} up={up} theme={theme} setTheme={setTheme}/>}
    </div>
  </div><Resizer side="left" value={w} onChange={setW} reset={()=>setW(292)}/></aside>;
}
