import { useRef, useLayoutEffect } from "react";
import { Icon } from "./Icon.jsx";
import { useI18n } from "../lib/i18n.jsx";

export const MAX=500;

export function Composer({mode,canAsk,text,setText,busy,send,stop}){
  const {t}=useI18n();
  const ref=useRef(null);
  const n=text.length,over=n>MAX;
  useLayoutEffect(()=>{const el=ref.current;if(!el)return;el.style.height="auto";el.style.height=Math.min(el.scrollHeight,150)+"px";},[text]); // ~6 lines
  return <div className="composer-wrap">
    <div className={"composer glass"+(over?" over":"")} style={{"--r":"30px"}}>
      <textarea ref={ref} aria-label={t("comp.label")} aria-invalid={over} rows="1" value={text} placeholder={t("comp.ph."+mode)} disabled={!canAsk}
        onChange={e=>setText(e.target.value)}
        onKeyDown={e=>{
          if(e.key==="Enter"&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();send();}
          else if(e.key==="Escape")e.currentTarget.blur();
        }}/>
      {busy?<button type="button" className="btn send stop" aria-label={t("comp.stop")} onClick={stop}><Icon n="stop"/></button>
      :<button type="button" className="btn send" aria-label={t("comp.send")} disabled={!canAsk||!text.trim()||over} onClick={()=>send()}><Icon n="send"/></button>}
    </div>
    <div className="cfoot">
      {canAsk?<span className="hint">{t("comp.hint")}</span>:<span className="warn" role="status">{t("comp.need."+mode)}</span>}
      {over&&<span className="bad" role="alert">{t("comp.over",{n:MAX})}</span>}
      <span className={over?"bad":n>=MAX*0.9?"warn":""} aria-label={n+" / "+MAX}>{n} / {MAX}</span>
    </div>
  </div>;
}
