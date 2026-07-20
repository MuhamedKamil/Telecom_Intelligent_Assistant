from pathlib import Path
from datetime import datetime
import mimetypes

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title


class DocumentFormatter:

    def format(self, file_path: str, chunks: list) -> dict:

        path = Path(file_path)

        mime_type, _ = mimetypes.guess_type(path)

        formatted_chunks = []
        current_position = 0
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):

            text = chunk.text.strip()

            start = current_position
            end = start + len(text)

            formatted_chunks.append({
                "chunk_id": f"chunk_{idx + 1:03d}",
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "text": text,
                "lang": chunk.metadata.languages[0] if chunk.metadata.languages else "unknown",
                "metadata": {
                    "page_number": getattr(chunk.metadata, "page_number", None),
                    "token_count": len(text.split()),
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "sentence_count": (
                        text.count(".")
                        + text.count("!")
                        + text.count("?")
                    ),
                    "start_position": start,
                    "end_position": end,
                },
            })

            current_position = end

        return {
            "document": {
                "source": "uploaded",
                "file_name": path.name,
                "file_type": mime_type,
                "last_modified": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat(),
                "total_chunks": total_chunks,
            },
            "chunks": formatted_chunks,
            "processing_metadata": {
                "chunking_strategy": "unstructured_chunk_by_title",
                "processed_at": datetime.utcnow().isoformat(),
            },
        }


class DocumentPipeline:

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
  
    }

    def __init__(
        self,
        max_characters=1000,
        new_after_n_chars=800,
        overlap=100,
    ):

        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.overlap = overlap

        self.formatter = DocumentFormatter()

    def process(self, file_path: str) -> dict:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}"
            )

        # Automatically detects the document type
        elements = partition(filename=str(path))

        chunks = chunk_by_title(
            elements,
            max_characters=self.max_characters,
            new_after_n_chars=self.new_after_n_chars,
            overlap=self.overlap,
        )

        return self.formatter.format(str(path), chunks)