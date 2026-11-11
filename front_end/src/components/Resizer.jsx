export const clampW=w=>Math.round(Math.max(240,Math.min(w,560,window.innerWidth*.45)));

/* drag handle on the inner edge of a side bar; side = "left" | "right", value/onChange in px */
export function Resizer({side,value,onChange,reset}){
  const down=e=>{
    if(e.button!==0)return;
    const bar=e.currentTarget.parentElement,el=e.currentTarget;
    el.setPointerCapture(e.pointerId);document.body.classList.add("resizing");
    const mv=ev=>{const r=bar.getBoundingClientRect();onChange(clampW(side==="left"?ev.clientX-r.left:r.right-ev.clientX));};
    const up=()=>{document.body.classList.remove("resizing");el.removeEventListener("pointermove",mv);el.removeEventListener("pointerup",up);el.removeEventListener("pointercancel",up);};
    el.addEventListener("pointermove",mv);el.addEventListener("pointerup",up);el.addEventListener("pointercancel",up);
  };
  const key=e=>{
    const d=e.key==="ArrowRight"?16:e.key==="ArrowLeft"?-16:0;
    if(d){e.preventDefault();onChange(clampW(value+(side==="left"?d:-d)));} // arrow follows the edge direction
  };
  return <div className={"resizer "+side} role="separator" aria-orientation="vertical" aria-valuenow={value} aria-valuemin={240} tabIndex={0} onPointerDown={down} onKeyDown={key} onDoubleClick={reset}/>;
}
