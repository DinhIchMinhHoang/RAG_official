import { useRef, useState } from "react";
import { Icon } from "./Icon.jsx";
import { useI18n } from "../lib/i18n.jsx";

export const ACCEPT=".pdf,.txt,.docx,.doc,.md,.tex,.pptx,.ppt";

export function Sources({corpus}){
  const {t}=useI18n();
  const {srcs,add,toggle,remove,open}=corpus;
  const inp=useRef(null),[over,setOver]=useState(false);
  return <div className="srcbox">
    <div className="shead"><span className="lb">{t("src.title")}</span>
      <button className="linkbtn" onClick={()=>inp.current.click()}><Icon n="plus"/>{t("src.add")}</button></div>
    <input ref={inp} type="file" multiple accept={ACCEPT} hidden onChange={e=>{add(e.target.files);e.target.value="";}}/>
    <div className={"drop"+(over?" over":"")}
      onDragOver={e=>{e.preventDefault();setOver(true);}} onDragLeave={()=>setOver(false)}
      onDrop={e=>{e.preventDefault();setOver(false);add(e.dataTransfer.files);}}>
      <Icon n="upload"/>{t("src.drop")}
    </div>
    <div className="srclist">
      {srcs.map(s=><div key={s.id} className="srcrow">
        {s.status==="ready"
          ?<input type="checkbox" checked={s.checked} aria-label={t("src.include",{name:s.name})} onChange={()=>toggle(s.id)}/>
          :<span className="spin" role="status" aria-label={t("src.indexing")}/>}
        <span className="ftype" data-t={s.type}>{s.type}</span>
        <button className="sname" title={s.name} onClick={()=>open(s.id)}>
          <span className="t">{s.name}</span>
          <span className={"m"+(s.status==="ready"?"":" busy")}>{t("src."+s.status)}</span>
        </button>
        <button className="iconbtn" aria-label={t("src.remove",{name:s.name})} onClick={()=>remove(s.id)}><Icon n="x"/></button>
      </div>)}
      {!srcs.length&&<div className="empty">{t("src.empty")}</div>}
    </div>
  </div>;
}
