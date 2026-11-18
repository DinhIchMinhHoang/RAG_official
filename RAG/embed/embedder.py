"""BGE-M3 dense embeddings; no E5 prefixes or word segmentation."""
import unicodedata

MODEL_NAME = "BAAI/bge-m3"


def clean_text(text):
    return unicodedata.normalize("NFC", text).strip()


def child_text(child):
    return clean_text(child["breadcrumb"]) + "\n" + clean_text(child["text"])


class Embedder:
    def __init__(self, device=None, max_length=1024, revision=None):
        from sentence_transformers import SentenceTransformer
        if not 8 <= max_length <= 8192:
            raise ValueError("max_length phải nằm trong [8, 8192].")
        self.model = SentenceTransformer(MODEL_NAME, device=device, revision=revision)
        self.model.max_seq_length = max_length
        self.max_length = max_length
        self.resolved_revision = getattr(self.model[0].auto_model.config, "_commit_hash", None)

    def token_lengths(self, texts):
        # Count the exact input, including special tokens, BEFORE encode truncation.
        return [len(ids) for ids in self.model.tokenizer(
            texts, padding=False, truncation=False, add_special_tokens=True
        )["input_ids"]]

    def encode(self, texts, batch_size=4, show_progress=True):
        if not texts or any(not clean_text(t) for t in texts):
            raise ValueError("Không thể embed danh sách rỗng hoặc text rỗng.")
        texts = [clean_text(t) for t in texts]
        lengths = self.token_lengths(texts)
        too_long = [(i, n) for i, n in enumerate(lengths) if n > self.max_length]
        if too_long:
            raise ValueError(
                f"Input vượt {self.max_length} token (vị trí, token): {too_long[:10]}. "
                "Tăng --max-length (tối đa 8192) hoặc chunk lại; không cắt ngầm."
            )
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
            prompt="",  # BGE-M3 does not require query/passage instructions.
        ).astype("float32")

    def encode_query(self, query):
        return self.encode([query], batch_size=1, show_progress=False)
