/* ---------- mock data ---------- */
const MIN=6e4,H=36e5,D=864e5,sod=new Date().setHours(0,0,0,0),now=Date.now();
export const HIST=[
  ["Who introduced BERT?","eqa",now-2*MIN],["What does the retriever find?","eqa",now-H],["When was BERT introduced?","eqa",sod-3*H],
  ["What is retrieval-augmented generation?","rag",now-14*MIN],["How does the generator use passages?","rag",now-3*H],["Why ground answers in a corpus?","rag",now-2*D]
].map(([src,mode,ts],i)=>({id:"h"+i,src,mode,ts,pinned:false}));
