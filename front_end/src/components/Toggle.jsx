export function Toggle({on,set,label}){return <button type="button" role="switch" aria-checked={on} aria-label={label} className={"sw"+(on?" on":"")} onClick={()=>set(!on)}/>;}
