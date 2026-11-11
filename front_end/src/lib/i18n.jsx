import { createContext, useContext, useEffect, useState } from "react";
import notch from "../i18n/notch.js";

// UI language (separate from the QA mode)
const own={
  en:{
    "side.toggle":"Toggle sidebar","insp.toggle":"Toggle reader","insp.title":"Evidence",
    "tab.history":"History","tab.settings":"Settings","mode.eqa":"Extractive QA","mode.rag":"RAG",
    "ctx.title":"Context","ctx.ph":"Paste a paragraph to ask questions about...","ctx.clear":"Clear context","ctx.count":"{n} chars",
    "src.title":"Sources","src.add":"Add source","src.drop":"or drop files here","src.indexing":"Indexing...","src.ready":"Ready",
    "src.include":"Include {name} in search","src.remove":"Remove {name}","src.empty":"No sources yet.",
    "hist.new":"New chat","hist.search":"Search history","hist.none":"No matches.","hist.empty":"No history yet.",
    "hist.pinned":"Pinned","hist.today":"Today","hist.yesterday":"Yesterday","hist.earlier":"Earlier",
    "hist.pin":"Pin","hist.unpin":"Unpin","hist.delete":"Delete","hist.deleted":"Deleted from history","hist.undo":"Undo",
    "set.topk":"Top-K results","set.thr":"Score threshold","set.reduce":"Reduce motion",
    "set.theme":"Theme","theme.dark":"Dark","theme.light":"Light","theme.auto":"Auto (system)","set.lang":"Interface language",
    "chat.start.eqa":"Ask a question about your context.","chat.start.rag":"Ask a question about your sources.","chat.thinking":"Thinking","chat.copy":"Copy answer",
    "chat.error":"Could not get an answer.","chat.retry":"Retry","chat.noans":"No answer above the score threshold.","chat.cite":"Citation {n}",
    "comp.label":"Question","comp.ph.eqa":"Ask a question about the context...","comp.ph.rag":"Ask a question about the sources...",
    "comp.need.eqa":"Add a context paragraph in the sidebar first.","comp.need.rag":"Add a source and wait until it is ready and checked.",
    "comp.send":"Send","comp.stop":"Stop generating","chat.stopped":"Stopped","comp.hint":"Enter to send · Shift+Enter new line · Ctrl/Cmd+Shift+L switch mode · Esc to leave",
    "comp.over":"Too long: limit is {n} characters.","net.offline":"You are offline. Answers may fail.",
    "ins.none":"Select an answer to see its evidence.","ins.score":"Confidence","ins.cands":"Candidates","ins.context":"Context",
    "ins.tab.source":"Source","ins.tab.corpus":"Corpus","ins.srcNone":"Click a citation [n] in an answer to see its source.",
    "ins.corpusNone":"Add sources to read them here.","ins.page":"Page {n}","ins.missing":"This source is no longer in the corpus."
  },
  vi:{
    "side.toggle":"Bật/tắt thanh bên","insp.toggle":"Bật/tắt bảng đọc","insp.title":"Bằng chứng",
    "tab.history":"Lịch sử","tab.settings":"Cài đặt","mode.eqa":"Extractive QA","mode.rag":"RAG",
    "ctx.title":"Ngữ cảnh","ctx.ph":"Dán một đoạn văn để đặt câu hỏi...","ctx.clear":"Xóa ngữ cảnh","ctx.count":"{n} ký tự",
    "src.title":"Nguồn","src.add":"Thêm nguồn","src.drop":"hoặc thả tệp vào đây","src.indexing":"Đang lập chỉ mục...","src.ready":"Sẵn sàng",
    "src.include":"Đưa {name} vào tìm kiếm","src.remove":"Xóa {name}","src.empty":"Chưa có nguồn nào.",
    "hist.new":"Cuộc trò chuyện mới","hist.search":"Tìm trong lịch sử","hist.none":"Không có kết quả.","hist.empty":"Chưa có lịch sử.",
    "hist.pinned":"Đã ghim","hist.today":"Hôm nay","hist.yesterday":"Hôm qua","hist.earlier":"Trước đó",
    "hist.pin":"Ghim","hist.unpin":"Bỏ ghim","hist.delete":"Xóa","hist.deleted":"Đã xóa khỏi lịch sử","hist.undo":"Hoàn tác",
    "set.topk":"Số kết quả Top-K","set.thr":"Ngưỡng điểm","set.reduce":"Giảm chuyển động",
    "set.theme":"Giao diện","theme.dark":"Tối","theme.light":"Sáng","theme.auto":"Tự động (theo hệ thống)","set.lang":"Ngôn ngữ giao diện",
    "chat.start.eqa":"Đặt câu hỏi về ngữ cảnh của bạn.","chat.start.rag":"Đặt câu hỏi về các nguồn của bạn.","chat.thinking":"Đang xử lý","chat.copy":"Sao chép câu trả lời",
    "chat.error":"Không lấy được câu trả lời.","chat.retry":"Thử lại","chat.noans":"Không có câu trả lời vượt ngưỡng điểm.","chat.cite":"Trích dẫn {n}",
    "comp.label":"Câu hỏi","comp.ph.eqa":"Đặt câu hỏi về ngữ cảnh...","comp.ph.rag":"Đặt câu hỏi về các nguồn...",
    "comp.need.eqa":"Hãy thêm đoạn ngữ cảnh ở thanh bên trước.","comp.need.rag":"Hãy thêm nguồn và chờ đến khi sẵn sàng và được chọn.",
    "comp.send":"Gửi","comp.stop":"Dừng tạo","chat.stopped":"Đã dừng","comp.hint":"Enter để gửi · Shift+Enter xuống dòng · Ctrl/Cmd+Shift+L đổi chế độ · Esc để thoát",
    "comp.over":"Quá dài: tối đa {n} ký tự.","net.offline":"Bạn đang ngoại tuyến. Có thể không trả lời được.",
    "ins.none":"Chọn một câu trả lời để xem bằng chứng.","ins.score":"Độ tin cậy","ins.cands":"Ứng viên","ins.context":"Ngữ cảnh",
    "ins.tab.source":"Nguồn","ins.tab.corpus":"Kho tài liệu","ins.srcNone":"Bấm vào trích dẫn [n] trong câu trả lời để xem nguồn.",
    "ins.corpusNone":"Thêm nguồn để đọc tại đây.","ins.page":"Trang {n}","ins.missing":"Nguồn này không còn trong kho tài liệu."
  }
};
const dict={en:{...own.en,...notch.en},vi:{...own.vi,...notch.vi}};
const KEY="lumen.ui";
const Ctx=createContext({t:k=>k,lang:"vi",setLang:()=>{}});

export function I18nProvider({children}){
  const [lang,setLangS]=useState(()=>{try{const v=localStorage.getItem(KEY);return v==="en"||v==="vi"?v:"vi";}catch{return "vi";}});
  const setLang=l=>{setLangS(l);try{localStorage.setItem(KEY,l);}catch{}};
  useEffect(()=>{document.documentElement.lang=lang;},[lang]);
  const t=(k,v)=>{let s=dict[lang][k]??dict.en[k]??k;if(v)for(const n in v)s=s.replace("{"+n+"}",v[n]);return s;};
  return <Ctx.Provider value={{t,lang,setLang}}>{children}</Ctx.Provider>;
}
export const useI18n=()=>useContext(Ctx);
