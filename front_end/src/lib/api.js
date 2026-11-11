/*
 * api.js: the ONLY producer of answers for the UI.
 *
 * Backend contract (replace each body with fetch() to a real QA service):
 *   askEqa({question, context, opts}) -> Promise<{
 *     answer:     string,   // extracted span, "" when nothing passes opts.threshold
 *     start, end: number,   // character offsets of the answer inside `context` (-1 when no answer)
 *     score:      number,   // confidence 0..1 of the top candidate
 *     candidates: {text, start, end, score}[],  // best first, at most opts.topK, all >= opts.threshold
 *     ms:         number    // server-side latency in ms
 *   }>
 *   askRag({question, sources, opts}) -> Promise<{
 *     answer:    string,    // text with inline markers "[1]" that point at citations[].id
 *     citations: {id, srcId, page, quote, score}[],  // id = 1-based marker number, page = 1-based, quote = verbatim text of that page
 *     ms:        number
 *   }>
 *   sources: [{id, name, type, pages: string[]}] (only ready + checked ones)
 *   opts: { topK, threshold }   // max results (1..10), minimum score (0..1)
 *   indexSource(file) -> Promise<{pages: string[]}>  // upload + index; resolves when the source is searchable
 *   streamEqa(args, {signal}) / streamRag(args, {signal}) -> AsyncGenerator of events (same args as askEqa/askRag):
 *     {type:"token", text}      // next chunk of `answer`; for RAG it includes the "[n]" markers
 *     {type:"done", result}     // LAST event: the full result object of askEqa/askRag
 *   Real backend: fetch(url,{signal}) + response.body ReadableStream (or SSE) emitting these same events;
 *   abort via `signal` must stop the stream (throw AbortError).
 * The current bodies are local mocks: word overlap between question and sentences.
 */
const STOP=new Set("a an the of in on at by to for and or is are was were be been do does did what who whom whose when where why how which that this with as from it its".split(" "));
const words=s=>[...s.matchAll(/[\p{L}\p{N}]+/gu)].map(m=>({w:m[0].toLowerCase(),i:m.index,e:m.index+m[0].length}));
const qset=q=>new Set(words(q).map(x=>x.w).filter(w=>!STOP.has(w)));
const sents=text=>[...text.matchAll(/[^.!?\n]+[.!?]*/g)].filter(m=>m[0].trim()).map(m=>{
  const lead=m[0].length-m[0].trimStart().length,text=m[0].trim();
  return {text,start:m.index+lead,end:m.index+lead+text.length};
});
const score=(s,qs)=>{const ov=new Set(words(s).map(x=>x.w).filter(w=>qs.has(w))).size;return ov?+Math.min(.99,.2+.79*ov/qs.size).toFixed(2):0;};
// best span: up to 4 content words right after the last question word, else the first run
const span=(s,qs)=>{
  const ws=words(s.text),hit=ws.map(x=>qs.has(x.w)),ok=x=>!STOP.has(x.w)&&!qs.has(x.w),from=hit.lastIndexOf(true)+1;
  let a=ws.findIndex((x,i)=>i>=from&&ok(x));if(a<0)a=ws.findIndex(ok);
  if(a<0)return {start:s.start,end:s.end};
  let b=a;while(b+1<ws.length&&b-a<3&&ok(ws[b+1]))b++;
  return {start:s.start+ws[a].i,end:s.start+ws[b].e};
};
const top=(xs,{topK=3,threshold=.3})=>xs.filter(x=>x.score>0&&x.score>=threshold).sort((a,b)=>b.score-a.score).slice(0,topK);

export async function askEqa({question,context,opts={}}){
  const qs=qset(question);
  const candidates=top(sents(context).map(s=>({s,score:score(s.text,qs)})),opts).map(({s,score})=>{const p=span(s,qs);return {text:context.slice(p.start,p.end),...p,score};});
  const c=candidates[0];
  return {answer:c?c.text:"",start:c?c.start:-1,end:c?c.end:-1,score:c?c.score:0,candidates,ms:60+context.length%90};
}

export async function askRag({question,sources,opts={}}){
  const qs=qset(question);
  const all=sources.flatMap(s=>s.pages.flatMap((p,pi)=>sents(p).map(x=>({srcId:s.id,page:pi+1,quote:x.text,score:score(x.text,qs)}))));
  const citations=top(all,opts).map((c,i)=>({id:i+1,...c}));
  return {answer:citations.map(c=>`${c.quote} [${c.id}]`).join(" "),citations,ms:120+question.length%80};
}

// mock streaming: compute the full result, then replay its answer word by word
async function* fake(run,signal){
  const nap=ms=>new Promise((ok,no)=>{
    if(signal&&signal.aborted)return no(signal.reason);
    const id=setTimeout(ok,ms);
    if(signal)signal.addEventListener("abort",()=>{clearTimeout(id);no(signal.reason);},{once:true});
  });
  await nap(300);
  const r=await run();
  for(const text of r.answer.match(/\S+\s*/g)||[]){yield {type:"token",text};await nap(20+Math.random()*20);}
  yield {type:"done",result:r};
}
export const streamEqa=(args,{signal}={})=>fake(()=>askEqa(args),signal);
export const streamRag=(args,{signal}={})=>fake(()=>askRag(args),signal);

export const extOf=name=>name.split(".").pop().toLowerCase();
const TEXT=["txt","md","tex"];
const readText=f=>new Promise((ok,no)=>{const r=new FileReader();r.onload=()=>ok(String(r.result));r.onerror=()=>no(r.error);r.readAsText(f);});
// shortcut: pages = ~1200-char paragraph groups; binary formats (pdf, docx...) get mock placeholder pages
export async function indexSource(file){
  const delay=new Promise(r=>setTimeout(r,800+file.size%700));
  let pages;
  if(TEXT.includes(extOf(file.name))){
    pages=[];let cur="";
    for(const p of (await readText(file)).split(/\n\s*\n/)){cur=cur?cur+"\n\n"+p:p;if(cur.length>1200){pages.push(cur);cur="";}}
    if(cur.trim())pages.push(cur);
    if(!pages.length)pages=[""];
  }else pages=[1,2,3].map(n=>`Placeholder page ${n} of ${file.name}. Preview is not available for this file type in the mock backend. Retrieval finds relevant passages from a corpus of documents.`);
  await delay;
  return {pages};
}
