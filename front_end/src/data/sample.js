/* ---------- mock data: starter context + corpus ---------- */
export const SAMPLE_CTX="Retrieval-augmented generation combines a retriever with a language model. The retriever finds relevant passages from a corpus. The generator then writes an answer grounded in those passages. Extractive question answering instead selects a span of text from a given context. BERT was introduced by Google in 2018 and set new records on SQuAD.";
export const SAMPLE_SRC={id:"s1",name:"rag-notes.md",type:"md",status:"ready",checked:true,pages:[
  "Retrieval-augmented generation is a technique that grounds answers in retrieved passages. The retriever finds relevant passages from a corpus. A dense retriever embeds the question and the passages into the same vector space.",
  "The generator reads the retrieved passages and writes an answer. Grounding answers in a corpus reduces hallucination. Each claim in the answer can be traced back to a cited source."
]};
