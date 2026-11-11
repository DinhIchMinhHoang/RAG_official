import { Toggle } from "./Toggle.jsx";
import { Select } from "./Select.jsx";
import { useI18n } from "../lib/i18n.jsx";

export function Settings({st,up,theme,setTheme}){
  const {t,lang,setLang}=useI18n();
  return <div>
    <div className="row"><div className="h"><label htmlFor="tk">{t("set.topk")}</label><span className="v">{st.topK}</span></div>
      <input type="range" id="tk" min={1} max={10} step={1} value={st.topK} onChange={e=>up("topK",+e.target.value)}/></div>
    <div className="row"><div className="h"><label htmlFor="thr">{t("set.thr")}</label><span className="v">{st.thr.toFixed(2)}</span></div>
      <input type="range" id="thr" min={0} max={1} step={0.05} value={st.thr} onChange={e=>up("thr",+e.target.value)}/></div>
    <div className="row"><div className="h"><span className="lb">{t("set.reduce")}</span><Toggle on={st.reduce} set={v=>up("reduce",v)} label={t("set.reduce")}/></div></div>
    <div className="row"><div className="h"><label htmlFor="th">{t("set.theme")}</label>
      <Select id="th" label={t("set.theme")} value={theme} onChange={setTheme} options={["dark","light","auto"].map(k=>({value:k,label:t("theme."+k)}))}/></div></div>
    <div className="row"><div className="h"><label htmlFor="ui">{t("set.lang")}</label>
      <Select id="ui" label={t("set.lang")} value={lang} onChange={setLang} options={[{value:"vi",label:"Tiếng Việt"},{value:"en",label:"English"}]}/></div></div>
  </div>;
}
