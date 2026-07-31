"""
Orchestrator: Voice-enabled RAG System
Combines ASR (Speech-to-Text), RAG pipeline, and TTS (Text-to-Speech).
"""

import time
import json
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from datetime import datetime
import glob

from ASR.asr_model import ASR
from RAG.rag import RAGSystem
from TTS.tts_model import TTS


class Orchestrator:
    """
    Main orchestrator for a voice-enabled RAG (Retrieval-Augmented Generation) system.
    
    This class integrates three core components:
        1. ASR (Automatic Speech Recognition): Converts speech to text
        2. RAG Pipeline: Retrieves relevant documents and generates responses
        3. TTS (Text-to-Speech): Converts responses back to speech
    
    The orchestrator manages document ingestion, multi-turn conversations,
    source tracking, and end-to-end voice interaction.
    """
    def __init__(
        self,
        # ASR
        asr_model_name: str = "base",
        asr_device: str = "cuda",
        asr_compute_type: str = "float32",
        # RAG
        embedder_model: str = "BAAI/bge-m3",
        llm_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        max_turns: int = 5,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
        # TTS
        tts_reference_audio: Optional[str] = None,
        tts_reference_text: Optional[str] = None,
        # Document Pipeline
        max_characters: int = 1000,
        new_after_n_chars: int = 800,
        overlap: int = 100,
        # General
        output_dir: str = "outputs",
        verbose: bool = True,
    ):
        """
        Initialize the Orchestrator with all necessary components.
        
        Args:
            # ASR Parameters
            asr_model_name (str): Whisper model size ("base", "small", "medium", "large").
                Defaults to "base".
            asr_device (str): Device for ASR ("cuda", "cpu"). Defaults to "cuda".
            asr_compute_type (str): Precision for ASR ("float32", "float16", "int8").
                Defaults to "float32".
            
            # RAG Parameters
            embedder_model (str): HuggingFace model for embeddings.
                Defaults to "BAAI/bge-m3".
            llm_model (str): HuggingFace model for response generation.
                Defaults to "meta-llama/Llama-3.2-1B-Instruct".
            max_turns (int): Maximum conversation turns to keep in context.
                Defaults to 5.
            top_k (int): Number of document chunks to retrieve per query.
                Defaults to 5.
            system_prompt (Optional[str]): Custom system prompt for LLM.
                If None, uses Telecom Egypt-specific default prompt.
            
            # TTS Parameters
            tts_reference_audio (Optional[str]): Path to reference audio for voice cloning.
                If None, TTS is disabled. Defaults to None.
            tts_reference_text (Optional[str]): Text corresponding to reference audio.
                Required if tts_reference_audio is provided.
            
            # Document Pipeline Parameters
            max_characters (int): Maximum characters per chunk. Defaults to 1000.
            new_after_n_chars (int): Characters before creating new chunk. Defaults to 800.
            overlap (int): Overlap characters between chunks. Defaults to 100.
            
            # General Parameters
            output_dir (str): Directory for saving outputs (audio, sessions).
                Defaults to "outputs".
            verbose (bool): Whether to print detailed logs. Defaults to True.
        
        Attributes:
            verbose (bool): Logging verbosity flag.
            output_dir (Path): Path to output directory.
            asr (ASR): Speech recognition component.
            rag (RAGSystem): RAG pipeline component.
            document_pipeline (Optional): Document processing pipeline.
            tts (Optional[TTS]): Text-to-speech component.
            conversation_id (int): Auto-incrementing ID for audio responses.
            source_history (List[Dict]): History of all loaded sources.
            total_chunks_loaded (int): Total number of chunks in system.
            document_mapping (Dict): Maps chunk IDs to source metadata.
        """
        self.verbose = verbose
        self._log("Initializing Orchestrator...")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ASR
        self._log("🎤 Initializing ASR...")
        self.asr = ASR(
            model_name=asr_model_name,
            device=asr_device,
            compute_type=asr_compute_type,
        )
        self._log("ASR initialized")

        # Initialize RAG with system prompt that encourages source use
        if system_prompt is None:
            system_prompt = """You are a helpful assistant for Telecom Egypt (WE).
Answer questions using ONLY the provided context.
If the answer is not in the context, say "I don't have this information in my knowledge base."
Always cite your sources.
Be concise and accurate.
Answer in the same language as the question."""

        self._log("Initializing RAG...")
        self.rag = RAGSystem(
            embedder_model=embedder_model,
            llm_model=llm_model,
            max_turns=max_turns,
            top_k=top_k,
            system_prompt=system_prompt,
        )
        self._log("RAG initialized")

        # Initialize Document Pipeline (with fallback)
        self._log("Initializing Document Pipeline...")
        self.document_pipeline = None
        self._init_document_pipeline(max_characters, new_after_n_chars, overlap)

        # Initialize TTS
        if tts_reference_audio and tts_reference_text:
            self._log("Initializing TTS...")
            self.tts = TTS(
                reference_audio=tts_reference_audio,
                reference_text=tts_reference_text,
            )
            self._log("TTS initialized with voice cloning")
        else:
            self.tts = None
            self._log(" TTS not initialized")

        # Tracking
        self.conversation_id = 0
        self.source_history = []
        self.total_chunks_loaded = 0
        self.document_mapping = {}

        self._log("Orchestrator initialized successfully!")

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _is_audio_file(self, path: str) -> bool:
        """Check if a file path is an audio file based on extension."""
        return Path(path).suffix.lower() in {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}

    def _init_document_pipeline(self, max_characters, new_after_n_chars, overlap):
        """
        Initialize document processing pipeline with fallback options.
        
        Attempts to use 'unstructured' library for advanced document parsing.
        If not available, uses a fallback pipeline.
        
        Args:
            max_characters (int): Maximum characters per chunk.
            new_after_n_chars (int): Characters before new chunk.
            overlap (int): Overlap between chunks.
        """        
        try:
            from unstructured.partition.auto import partition
            from unstructured.chunking.title import chunk_by_title

            class DocumentPipeline:
                def __init__(self, max_characters, new_after_n_chars, overlap):
                    self.max_characters = max_characters
                    self.new_after_n_chars = new_after_n_chars
                    self.overlap = overlap

                def process(self, file_path: str) -> Dict:
                    elements = partition(filename=str(file_path))
                    chunks = chunk_by_title(
                        elements,
                        max_characters=self.max_characters,
                        new_after_n_chars=self.new_after_n_chars,
                        overlap=self.overlap,
                    )

                    formatted_chunks = []
                    for idx, chunk in enumerate(chunks):
                        text = chunk.text.strip()
                        formatted_chunks.append({
                            "chunk_id": f"chunk_{idx + 1:03d}",
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "text": text,
                            "lang": getattr(chunk.metadata, "languages", ["unknown"])[0] if chunk.metadata else "unknown",
                            "metadata": {
                                "page_number": getattr(chunk.metadata, "page_number", None),
                                "token_count": len(text.split()),
                            },
                        })

                    return {
                        "document": {
                            "source": "uploaded",
                            "file_name": Path(file_path).name,
                            "file_type": Path(file_path).suffix,
                            "total_chunks": len(formatted_chunks),
                        },
                        "chunks": formatted_chunks,
                    }

            self.document_pipeline = DocumentPipeline(max_characters, new_after_n_chars, overlap)
            self._log("Document Pipeline initialized with unstructured")

        except ImportError:
            self._log("Unstructured not available - using fallback")
            self._log("Using fallback document pipeline")

    def _load_json(self, file_path: str) -> Dict:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _prepare_chunks_for_rag(self, chunks: List[Dict], doc_meta: Dict) -> List[Dict]:
        """
        Prepare document chunks for ingestion into RAG system.
        
        Args:
            chunks (List[Dict]): Raw chunks from document processing.
            doc_meta (Dict): Document metadata (title, url, source, etc.).
            
        Returns:
            List[Dict]: Formatted chunks ready for RAG ingestion.
        """
        rag_chunks = []
        doc_title = doc_meta.get("title", "Website Document")
        doc_url = doc_meta.get("url", "unknown")
        doc_source = doc_meta.get("sources", "website")
        doc_lang = doc_meta.get("lang", "en")

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if text and len(text.strip()) > 0:
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    chunk_id = f"{doc_source}_{idx+1:03d}"
                
                rag_chunks.append({
                    "text": text,
                    "source": "website",
                    "url": doc_url,
                    "title": doc_title,
                    "source_name": doc_source,
                    "language": doc_lang,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk.get("chunk_index", idx),
                    "total_chunks": len(chunks),
                    "metadata": chunk.get("metadata", {}),
                })

        return rag_chunks

    def add_web_data(self, source: Union[str, List[str], List[Dict], Dict]) -> None:
        """
        Add website data from various sources (JSON files or dictionaries).
        
        Supports multiple input formats:
            - Single JSON file path
            - Directory path containing JSON files
            - Glob pattern (e.g., "*.json")
            - List of file paths or dictionaries
            - Single dictionary with document data
        
        All documents are loaded in one batch to optimize embedding generation.
        
        Args:
            source (Union[str, List[str], List[Dict], Dict]): Source data to add.
            
        """
        # Normalize to list
        if isinstance(source, dict):
            sources = [source]
        elif isinstance(source, str):
            if Path(source).is_dir():
                sources = glob.glob(str(Path(source) / "*.json"))
            elif '*' in source:
                sources = glob.glob(source)
            else:
                sources = [source]
        elif isinstance(source, list):
            sources = source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        total_chunks = 0
        total_files = 0
        all_rag_chunks = []

        for item in sources:
            try:
                if isinstance(item, str):
                    self._log(f"Loading: {Path(item).name}")
                    web_data = self._load_json(item)
                else:
                    web_data = item

                doc_meta = web_data.get("original_document", {})
                chunks = web_data.get("chunks", [])

                if not chunks:
                    self._log(f"No chunks found in {item}")
                    continue

                rag_chunks = self._prepare_chunks_for_rag(chunks, doc_meta)

                if not rag_chunks:
                    self._log(f"No valid chunks in {item}")
                    continue

                doc_title = doc_meta.get("title", "Website Document")
                source_info = {
                    "type": "website",
                    "url": doc_meta.get("url", "unknown"),
                    "title": doc_title,
                    "source": doc_meta.get("sources", "website"),
                    "chunks": len(rag_chunks),
                    "file": Path(item).name if isinstance(item, str) else "direct",
                    "added_at": datetime.now().isoformat(),
                }
                self.source_history.append(source_info)

                all_rag_chunks.extend(rag_chunks)
                total_chunks += len(rag_chunks)
                total_files += 1
                self._log(f"Prepared {len(rag_chunks)} chunks from: {doc_title[:50]}...")

            except Exception as e:
                self._log(f"Failed to load {item}: {e}")

        # Load ALL chunks in ONE batch
        if all_rag_chunks:
            self.rag.add_website_documents(all_rag_chunks)
            self.total_chunks_loaded += len(all_rag_chunks)
            
            for chunk in all_rag_chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id:
                    for source_info in self.source_history:
                        if source_info.get("title") == chunk.get("title"):
                            self.document_mapping[chunk_id] = source_info
                            break

            self._log(f"\nBatch loaded {len(all_rag_chunks)} chunks from {total_files} file(s)")

        self._log(f"\nSummary: Loaded {total_chunks} chunks from {total_files} file(s)")
        self._log(f"Total chunks in system: {self.total_chunks_loaded}")

    def add_uploaded_file(self, file_path: str) -> None:
        """
        Add a single uploaded document file (PDF, DOCX, TXT, etc.) to the system.
        
        Uses the document pipeline to extract and chunk the document content.
        Supports various formats through the unstructured library.
        
        Args:
            file_path (str): Path to the document file.
            
        """
        if self.document_pipeline is None:
            self._log(" Document pipeline not available")
            return

        try:
            self._log(f"Processing: {Path(file_path).name}")
            result = self.document_pipeline.process(file_path)

            doc_meta = result.get("document", {})
            chunks = result.get("chunks", [])

            if not chunks:
                self._log("No chunks extracted from document")
                return

            file_name = doc_meta.get("file_name", "Unknown")
            file_type = doc_meta.get("file_type", "unknown")

            rag_chunks = []
            for chunk in chunks:
                text = chunk.get("text", "")
                if text and len(text.strip()) > 0:
                    rag_chunks.append({
                        "text": text,
                        "source": "uploaded",
                        "file_name": file_name,
                        "file_type": file_type,
                        "chunk_id": chunk.get("chunk_id", f"uploaded_{len(rag_chunks)+1:03d}"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "total_chunks": len(chunks),
                        "metadata": chunk.get("metadata", {}),
                        "lang": chunk.get("lang", "unknown"),
                    })

            if not rag_chunks:
                self._log("No valid chunks from document")
                return

            document_metadata = {
                "title": file_name,
                "source": "uploaded",
                "file_type": file_type,
                "total_chunks": len(rag_chunks),
            }

            self.rag.add_uploaded_documents(document_metadata, rag_chunks)
            self.total_chunks_loaded += len(rag_chunks)

            source_info = {
                "type": "uploaded",
                "file_name": file_name,
                "file_type": file_type,
                "chunks": len(rag_chunks),
                "added_at": datetime.now().isoformat(),
            }
            self.source_history.append(source_info)

            for chunk in rag_chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id:
                    self.document_mapping[chunk_id] = source_info

            self._log(f"Added {len(rag_chunks)} chunks from: {file_name}")

        except Exception as e:
            self._log(f"Failed to process file: {e}")

    # ============ Processing ============

    def process(self, input_data: str, return_audio: bool = True) -> Dict:
        """
        Main entry point for processing user input (text or audio).
        
        Automatically detects input type and routes to appropriate processor.
        
        Args:
            input_data (str): Text question or path to audio file.
            return_audio (bool): Whether to generate audio response. Defaults to True.
            
        Returns:
            Dict: Processing results containing:
                - input_type (str): "text" or "audio"
                - question (str): The processed question
                - response (str): Generated answer
                - sources (List[Dict]): Sources used for answer
                - sources_text (str): Formatted sources for display
                - elapsed_time (float): Processing time in seconds
                - audio_path (Optional[str]): Path to generated audio (if TTS enabled)
                - language (str): Detected language (for audio input)
                - confidence (float): ASR confidence score (for audio input)
                
        """
        is_audio = self._is_audio_file(input_data) and Path(input_data).exists()

        self._log("=" * 60)
        self._log(f"{'AUDIO' if is_audio else 'TEXT'} INPUT")
        self._log(f" Total chunks in system: {self.total_chunks_loaded}")
        self._log("=" * 60)

        if is_audio:
            return self._process_audio(input_data, return_audio)
        return self._process_text(input_data)

    def _process_text(self, question: str) -> Dict:
        """
        Process text input through the RAG pipeline.
        
        Args:
            question (str): User's question text.
            
        Returns:
            Dict: Processing results (see process() for structure).
        """
        start = time.time()

        if self.total_chunks_loaded == 0:
            self._log("No documents loaded! Please add documents first.")
            return {
                "input_type": "text",
                "question": question,
                "response": "No documents loaded. Please add documents to the knowledge base first.",
                "sources": [],
                "sources_text": "No documents available.",
                "elapsed_time": time.time() - start,
                "audio_path": None,
            }

        rag_result = self.rag.ask_with_sources(question)
        
        # Enrich sources with full metadata from our mapping
        enriched_sources = []
        for src in rag_result.get("sources", []):
            chunk_id = src.get("chunk_id", "")
            if chunk_id in self.document_mapping:
                stored = self.document_mapping[chunk_id]
                enriched_src = {**src, **stored}
                enriched_sources.append(enriched_src)
            else:
                # Try to find by title match
                title = src.get("title", "")
                if title:
                    for source_info in self.source_history:
                        if source_info.get("title") == title:
                            enriched_src = {**src, **source_info}
                            enriched_sources.append(enriched_src)
                            break
                    else:
                        enriched_sources.append(src)
                else:
                    enriched_sources.append(src)

        sources_text = self._format_sources(enriched_sources)

        self._log(f"Q: {question[:100]}...")
        self._log(f"A: {rag_result['response'][:200]}...")
        self._log(f"Sources used: {len(enriched_sources)}")

        return {
            "input_type": "text",
            "question": question,
            "response": rag_result["response"],
            "sources": enriched_sources,
            "sources_text": sources_text,
            "elapsed_time": time.time() - start,
            "audio_path": None,
        }

    def _process_audio(self, audio_path: str, return_audio: bool) -> Dict:
        """
        Process audio input through ASR → RAG → (optional) TTS pipeline.
        
        Args:
            audio_path (str): Path to audio file.
            return_audio (bool): Whether to generate audio response.
            
        Returns:
            Dict: Processing results (see process() for structure).
        """
        start = time.time()

        if self.total_chunks_loaded == 0:
            self._log("No documents loaded! Please add documents first.")
            return {
                "input_type": "audio",
                "question": "No documents loaded",
                "response": "No documents loaded. Please add documents to the knowledge base first.",
                "sources": [],
                "sources_text": "No documents available.",
                "language": "unknown",
                "confidence": 0,
                "audio_path": None,
                "elapsed_time": time.time() - start,
            }

        asr_result = self.asr.transcribe(audio_path)
        question = asr_result["text"]


        # question = asr_result['text']


        self._log(f"Transcribed: {question[:100]}...")
        self._log(f"   Language: {asr_result.get('language', 'en')}")

        rag_result = self.rag.ask_with_sources(question)
        
        enriched_sources = []
        for src in rag_result.get("sources", []):
            chunk_id = src.get("chunk_id", "")
            if chunk_id in self.document_mapping:
                stored = self.document_mapping[chunk_id]
                enriched_src = {**src, **stored}
                enriched_sources.append(enriched_src)
            else:
                title = src.get("title", "")
                if title:
                    for source_info in self.source_history:
                        if source_info.get("title") == title:
                            enriched_src = {**src, **source_info}
                            enriched_sources.append(enriched_src)
                            break
                    else:
                        enriched_sources.append(src)
                else:
                    enriched_sources.append(src)

        response = rag_result["response"]

        audio_path_out = None
        if return_audio and self.tts:
            self.conversation_id += 1
            audio_path_out = self.output_dir / f"response_{self.conversation_id:04d}.wav"
            tts_result = self.tts.generate(text=response, output_file=str(audio_path_out))
            self._log(f"Audio: {audio_path_out}")
        elif return_audio and not self.tts:
            self._log("TTS not available")

        sources_text = self._format_sources(enriched_sources)

        self._log(f"A: {response[:200]}...")
        self._log(f"Sources used: {len(enriched_sources)}")

        return {
            "input_type": "audio",
            "question": question,
            "response": response,
            "sources": enriched_sources,
            "sources_text": sources_text,
            "language": asr_result.get('language', 'en'),
            "confidence": asr_result.get('language_probability', 0),
            "audio_path": str(audio_path_out) if audio_path_out else None,
            "elapsed_time": time.time() - start,
        }

    # ============ Formatting ============

    def _format_sources(self, sources: List[Dict]) -> str:
        """
        Format sources for human-readable display.
        
        Args:
            sources (List[Dict]): List of source dictionaries.
            
        Returns:
            str: Formatted source string.
        """
        if not sources:
            return "No sources available."

        formatted = []
        seen_titles = set()
        
        for i, src in enumerate(sources, 1):
            title = src.get('title', '')
            if title in seen_titles and len(sources) > 1:
                continue
            seen_titles.add(title)
            
            if src.get("source") == "website":
                url = src.get('url', 'N/A')
                chunk_id = src.get('chunk_id', '')
                chunk_info = f" (Chunk: {chunk_id})" if chunk_id else ""
                formatted.append(f"{i}. {title}{chunk_info}\n   URL: {url}")
            else:
                file_name = src.get('file_name', 'Unknown')
                chunk_id = src.get('chunk_id', '')
                chunk_info = f" (Chunk: {chunk_id})" if chunk_id else ""
                formatted.append(f"{i}.  {file_name}{chunk_info}")

        return "\n".join(formatted)

    # ============ Source Info ============

    def get_sources(self) -> Dict:
        """
        Get a summary of all loaded sources.
        
        Returns:
            Dict: Source summary containing:
                - total_sources (int): Total number of sources
                - website_sources (int): Number of website sources
                - uploaded_sources (int): Number of uploaded sources
                - total_chunks (int): Total chunks across all sources
                - details (List[Dict]): Detailed source information
        
        """
        web_count = sum(1 for s in self.source_history if s["type"] == "website")
        upload_count = sum(1 for s in self.source_history if s["type"] == "uploaded")
        total_chunks = sum(s.get("chunks", 0) for s in self.source_history)

        return {
            "total_sources": len(self.source_history),
            "website_sources": web_count,
            "uploaded_sources": upload_count,
            "total_chunks": total_chunks,
            "details": self.source_history,
        }

    # ============ Conversation Management ============

    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.rag.get_history()

    def clear_memory(self):
        """Clear conversation memory."""
        self.rag.clear_memory()
        self._log("Memory cleared")

    def reset(self):
        """Reset everything."""
        self.rag.clear()
        self.source_history.clear()
        self.document_mapping.clear()
        self.conversation_id = 0
        self.total_chunks_loaded = 0
        self._log("System reset")

    # ============ Save ============

    def save(self, output_path: Optional[str] = None):
        """
        Save the current session including conversation and source history.
        
        Args:
            output_path (Optional[str]): Path to save JSON file.
                If None, saves to output_dir/session.json.
        """
        if output_path is None:
            output_path = self.output_dir / "session.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "conversation": self.get_history(),
            "sources": self.source_history,
            "source_summary": self.get_sources(),
            "total_chunks_loaded": self.total_chunks_loaded,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._log(f"Saved session: {output_path}")

    def debug_documents(self):
        """
        Print debug information about loaded documents.
        
        Useful for troubleshooting and verifying document ingestion.
        """
        summary = self.get_sources()
        print("\n" + "=" * 60)
        print("DOCUMENT DEBUG INFO")
        print("=" * 60)
        print(f"Total chunks in system: {self.total_chunks_loaded}")
        print(f"Total sources: {summary['total_sources']}")
        print(f"  - Website sources: {summary['website_sources']}")
        print(f"  - Uploaded sources: {summary['uploaded_sources']}")
        print(f"Total chunks: {summary['total_chunks']}")
        
        if self.source_history:
            print("\nSource details (first 5):")
            for src in self.source_history[:5]:
                if src['type'] == 'website':
                    print(f"  {src.get('title', 'Untitled')} - {src.get('chunks', 0)} chunks")
                    print(f"     URL: {src.get('url', 'N/A')}")
                else:
                    print(f"  {src.get('file_name', 'Unknown')} - {src.get('chunks', 0)} chunks")
            if len(self.source_history) > 5:
                print(f"  ... and {len(self.source_history) - 5} more")
        else:
            print("\nNo sources loaded!")
        print(f"\nDocument mapping size: {len(self.document_mapping)}")
        print("=" * 60)

    def __repr__(self):
        return f"Orchestrator(docs={self.total_chunks_loaded}, turns={len(self.rag.memory)}, sources={len(self.source_history)})"