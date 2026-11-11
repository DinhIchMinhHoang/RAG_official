import { useState, useEffect } from "react";
import { HIST } from "../data/history.js";

const KEY="lumen.history.qa";
const load=()=>{try{const v=JSON.parse(localStorage.getItem(KEY));if(Array.isArray(v))return v.filter(h=>h&&typeof h.src==="string"&&(h.mode==="eqa"||h.mode==="rag"));}catch{}return HIST;};

// items are sessions: {id, src (first question), mode, ts, pinned, msgs (finished messages), ctx (EQA context)}; `undo` holds the last deleted item for the toast
export function useHistory(){
  const [items,setItems]=useState(load),[undo,setUndo]=useState(null);
  useEffect(()=>{try{localStorage.setItem(KEY,JSON.stringify(items));}catch{}},[items]);
  useEffect(()=>{if(!undo)return;const id=setTimeout(()=>setUndo(null),5000);return()=>clearTimeout(id);},[undo]);
  // upsert one finished message into session `sid` (created on first message)
  const add=(sid,mode,msg,ctx)=>setItems(l=>{
    const ts=Date.now(),put=h=>({...h,ts,ctx,msgs:[...(h.msgs||[]).filter(x=>x.id!==msg.id),msg]});
    return l.some(h=>h.id===sid)?l.map(h=>h.id===sid?put(h):h):[put({id:sid,src:msg.q,mode,pinned:false}),...l];
  });
  const pin=id=>setItems(l=>l.map(h=>h.id===id?{...h,pinned:!h.pinned}:h));
  const remove=id=>{const i=items.findIndex(h=>h.id===id);if(i<0)return;setUndo({item:items[i],i});setItems(l=>l.filter(h=>h.id!==id));};
  const restore=()=>{if(!undo)return;const {item,i}=undo;setItems(l=>{const c=[...l];c.splice(Math.min(i,c.length),0,item);return c;});setUndo(null);};
  return {items,add,pin,remove,undo,restore};
}
