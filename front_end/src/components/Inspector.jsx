import { Seg } from "./Seg.jsx";
import { EqaView } from "./EqaView.jsx";
import { SourceView } from "./SourceView.jsx";
import { CorpusView } from "./CorpusView.jsx";
import { useI18n } from "../lib/i18n.jsx";

/* right panel: eqa = answer in context; rag = Source | Corpus tabs. `ex` is the selected finished message of the current mode */
export function Inspector({mode,ex,cand,setCand,cite,setCite,tab,setTab,srcs,viewSrc,setViewSrc}){
  const {t}=useI18n();
  if(mode==="eqa")return <EqaView ex={ex} cand={cand} setCand={setCand}/>;
  return <>
    <Seg className="seg" aria-label={t("ins.tab.source")+" / "+t("ins.tab.corpus")}value={tab} onValueChange={setTab} items={["source","corpus"].map(k=>({value:k,label:t("ins.tab."+k)}))}/>
    {tab==="source"?<SourceView ex={ex} cite={cite} setCite={setCite} srcs={srcs}/>:<CorpusView srcs={srcs} viewSrc={viewSrc} setViewSrc={setViewSrc}/>}
  </>;
}
