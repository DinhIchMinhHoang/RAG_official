export function Seg({className,items,value,onValueChange,...rest}){
  const i=items.findIndex(o=>o.value===value);
  return <div className={className+(i>0?" s2":"")} role="tablist" {...rest}>
    <span className="thumb" aria-hidden="true"/>
    {items.map(o=><button key={o.value} type="button" role="tab" aria-selected={o.value===value} className={o.value===value?"on":""} onClick={()=>onValueChange(o.value)}>{o.label}</button>)}
  </div>;
}
