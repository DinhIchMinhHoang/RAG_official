import { useRef, useEffect } from "react";

/* text with [start,end) wrapped in <mark>, scrolled into view */
export function Hl({text,start,end}){
  const ref=useRef(null);
  useEffect(()=>{ref.current&&ref.current.scrollIntoView({block:"nearest",behavior:"smooth"});},[text,start,end]);
  if(start<0||end<=start)return <>{text}</>;
  return <>{text.slice(0,start)}<mark ref={ref}>{text.slice(start,end)}</mark>{text.slice(end)}</>;
}
