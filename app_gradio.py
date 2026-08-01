"""
Complete Voice-Enabled RAG App - Gradio Version
- Text Input → Text Output
- Audio Input → Audio Output
- Added chunk debugging
"""

import gradio as gr
import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Orchestration.orchestrator import Orchestrator


class AppState:
    """
    Application state container for the Gradio interface.
    
    This class maintains all global state across the web application,
    including the orchestrator instance, initialization status, logs,
    conversation history, and tracking metrics.
    
    Attributes:
        orchestrator (Orchestrator): The main RAG orchestrator instance.
        initialized (bool): Whether the system has been initialized.
        total_chunks (int): Total number of document chunks loaded.
        source_count (int): Number of document sources loaded.
        init_logs (List[str]): Log messages from initialization and operations.
        history (List[Dict]): Conversation history with user/assistant turns.
        conversation_id (int): Auto-incrementing ID for tracking turns.
    """
    def __init__(self):
        self.orchestrator = None
        self.initialized = False
        self.total_chunks = 0
        self.source_count = 0
        self.init_logs = []
        self.history = []
        self.conversation_id = 0

state = AppState()


def log_message(msg):
    """
    Log a message with timestamp and store in application state.
    
    Args:
        msg (str): Message to log.
        
    Returns:
        str: The formatted log entry with timestamp.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    state.init_logs.append(log_entry)
    print(log_entry)
    return log_entry


def initialize_system():
    """
    Initialize the RAG system with documents from the documents directory.
    
    This function:
        1. Scans the 'documents' folder for JSON files
        2. Loads and validates document data
        3. Initializes the Orchestrator with all components
        4. Loads documents into the RAG system
        5. Performs debug verification of loaded chunks
    
    Returns:
        str: Status message indicating initialization success or failure.
    """
    log_message("🚀 Starting Orchestrator Initialization...")
    
    docs_dir = Path("documents")
    if not docs_dir.exists():
        docs_dir.mkdir(exist_ok=True)
    
    json_files = list(docs_dir.glob("*.json"))
    log_message(f"Found {len(json_files)} JSON files")
    
    # DEBUG: Show all files found
    for f in json_files:
        log_message(f"  {f.name}")
    
    try:
        log_message("🔧 Initializing Orchestrator...")
        
        orchestrator = Orchestrator(
            embedder_model="BAAI/bge-m3",
            llm_model="meta-llama/Llama-3.2-3B-Instruct",
            max_turns=5,
            top_k=5,
            verbose=True,  # Enable verbose to see loading
            system_prompt="""You are a precise, factual assistant for Telecom Egypt (WE). Your purpose is to answer user questions accurately using ONLY the provided context.

══════════════════════════════════════════════════════════════════
CORE PRINCIPLES (MUST FOLLOW)
══════════════════════════════════════════════════════════════════

1. STRICT GROUNDING
   - Answer ONLY using information explicitly present in the context
   - NEVER add, infer, guess, or invent information not in the context
   - If information is not in the context, say: "I don't have this information in my knowledge base."

2. EXACT EXTRACTION
   - Extract information exactly as it appears in the context
   - For numbers, prices, dates, names: preserve the exact values
   - For lists: only include items explicitly mentioned in the context
   - For descriptions: paraphrase while preserving all key facts

3. STRUCTURED RESPONSES
   - When the context contains structured information (lists, categories, steps):
     - Preserve the structure
     - Include all items mentioned
     - Do NOT add items not mentioned
   - For questions asking "what", "how", "why": provide complete information
   - For questions asking "yes/no": answer directly with supporting evidence

4. HANDLE ALL INFORMATION TYPES
   - Prices/Costs: Extract exact numbers with currency (جنيه, EGP, $)
   - Dates/Times: Preserve exact format (e.g., "1:30 PM", "2024-01-01")
   - Names/Titles: Preserve exact spelling and formatting
   - Descriptions: Include all key attributes and characteristics
   - Steps/Procedures: Maintain the exact order and sequence
   - Comparisons: Include all differences and similarities mentioned

5. SOURCE ATTRIBUTION
   - When available, mention the source title or URL
   - Cite specific information confidently when present
   - Do NOT fabricate source details

6. LANGUAGE CONSISTENCY
   - Respond in the SAME language as the user's question
   - Arabic question → Arabic response
   - English question → English response

7. CONCISENESS
   - Be direct and to the point
   - Avoid unnecessary introductory phrases
   - Do not repeat the same information multiple times
   - For short answers, keep them brief but complete

8. HANDLE AMBIGUITY
   - If the question is unclear, ask for clarification
   - If multiple interpretations exist, state the one you're using
   - If the context contains conflicting information, mention the conflict

9. NEGATIVE CASES
   - If the context mentions something is NOT available, state this
   - If the answer would be "no", say so clearly
   - If the context is insufficient, state what IS known and what IS NOT

10. QUALITY STANDARDS
    - Every claim must be traceable to the context
    - Every number, date, and name must be verifiable
    - Every list item must appear in the context
    - Every instruction must be followed precisely

══════════════════════════════════════════════════════════════════
SPECIAL HANDLING FOR COMMON INFORMATION TYPES
══════════════════════════════════════════════════════════════════

For PRICES:
- Look for patterns like: "سعر", "تكلفة", "قيمة", "بسعر", "EGP", "جنيه"
- Extract: [number] + [currency] + [unit if applicable]
- Format: "السعر هو X جنيه" or "The price is X EGP"

For LISTS:
- Extract ONLY the items explicitly listed
- Preserve the exact order if meaningful
- Do NOT group or categorize differently than the context

For STEPS/PROCEDURES:
- Preserve the exact sequence
- Include all steps mentioned
- Do NOT add steps not mentioned

For DEFINITIONS:
- Use the exact definition from the context
- Include all parts of the definition
- Do NOT simplify or modify the definition

For COMPARISONS:
- Include all points of comparison mentioned
- Preserve the exact differences stated
- Do NOT add comparative analysis not in the context

══════════════════════════════════════════════════════════════════
EXAMPLES OF CORRECT BEHAVIOR
══════════════════════════════════════════════════════════════════

Context: "خدمة 140 دليل تقدم معلومات عن: العناوين، أرقام الاتصال، الأقسام المتاحة."
Question: "ما المعلومات التي تقدمها خدمة 140 دليل؟"
CORRECT: "خدمة 140 دليل تقدم معلومات عن: العناوين، أرقام الاتصال، الأقسام المتاحة."
WRONG: "تقدم خدمة 140 دليل معلومات عن الجامعات، المدارس، الخدمات الطبية..." (adding items not in context)

══════════════════════════════════════════════════════════════════

Context: "سعر الخدمة: 1.5 جنيه/الدقيقة"
Question: "كم سعر الخدمة؟"
CORRECT: "سعر الخدمة هو 1.5 جنيه في الدقيقة."
WRONG: "سعر الخدمة حوالي 2 جنيه" (changing the number)

══════════════════════════════════════════════════════════════════

Context: "الجامعات الحكومية والخاصة فقط مسموح لها بالتقديم."
Question: "ما هي الجامعات المسموح لها بالتقديم؟"
CORRECT: "الجامعات الحكومية والخاصة فقط مسموح لها بالتقديم."
WRONG: "الجامعات الحكومية والخاصة والأهلية مسموح لها بالتقديم." (adding "الأهلية")

══════════════════════════════════════════════════════════════════

Context: No information about mobile prices
Question: "ما هي أسعار باقات الموبايل؟"
CORRECT: "I don't have this information in my knowledge base."
WRONG: "أسعار الباقات تبدأ من 50 جنيه" (guessing)

══════════════════════════════════════════════════════════════════

REMEMBER: Your job is to be a faithful conveyor of information from the context to the user, not to interpret, expand, or add to the information provided. Accuracy and truthfulness are paramount.""",
            tts_reference_audio="reference_audio.wav",
            tts_reference_text="ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
        )
        
        documents = []
        total_docs = 0
        
        if json_files:
            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "original_document" in data and "chunks" in data:
                            documents.append(data)
                            total_docs += 1
                            chunk_count = len(data["chunks"])
                            doc_title = data["original_document"].get("title", "Unknown")
                            log_message(f"  {file_path.name} → {chunk_count} chunks - {doc_title[:50]}...")
                            
                            # DEBUG: Check for 140 Guide
                            if "140" in file_path.name.lower() or "140" in doc_title:
                                log_message(f"  🔍 FOUND 140 GUIDE DOCUMENT!")
                                for i, chunk in enumerate(data["chunks"]):
                                    text_preview = chunk.get("text", "")[:150]
                                    log_message(f"    Chunk {i+1}: {text_preview}...")
                except Exception as e:
                    log_message(f"  ❌ Error: {file_path.name}: {e}")
            
            if documents:
                log_message(f"📤 Loading {len(documents)} documents into RAG...")
                orchestrator.add_web_data(documents)
                state.total_chunks = orchestrator.total_chunks_loaded
                state.source_count = len(orchestrator.source_history)
                log_message(f"Loaded {state.total_chunks} chunks from {len(documents)} documents")
                
                # DEBUG: Check chunks in store
                log_message(f"Verifying chunks in store...")
                chunks = orchestrator.rag.website_store.chunks
                log_message(f"  Total chunks in store: {len(chunks)}")
                for i, chunk in enumerate(chunks[:5]):
                    title = chunk.get('title', 'Unknown')[:40]
                    text = chunk.get('text', '')[:100]
                    log_message(f"  Chunk {i+1}: {title} - {text}...")
                
                # DEBUG: Specifically check for 140 Guide chunks
                log_message(f"🔍 Looking for 140 Guide chunks...")
                found_140 = False
                for chunk in chunks:
                    title = chunk.get('title', '')
                    if "140" in title.lower():
                        found_140 = True
                        text = chunk.get('text', '')[:150]
                        log_message(f"  Found 140 Guide chunk: {text}...")
                if not found_140:
                    log_message(f"   No 140 Guide chunks found in store!")
        
        state.orchestrator = orchestrator
        state.initialized = True
        state.history = []
        state.conversation_id = 0
        
        log_message("🎉 System ready!")
        return f" System Ready!\nChunks: {state.total_chunks}\nSources: {state.source_count}"
        
    except Exception as e:
        log_message(f"❌ Error: {e}")
        import traceback
        log_message(traceback.format_exc())
        return f"❌ Failed: {str(e)}"


def process_text_query(question):
    """
    Process a text query through the RAG pipeline with optional audio output.
    
    Args:
        question (str): The user's question text.
        
    Returns:
        tuple: (response_text, audio_filepath) where audio_filepath may be None.
    """
    if not state.initialized or state.orchestrator is None:
        return "System not initialized.", None
    
    if state.orchestrator.total_chunks_loaded == 0:
        return "No documents loaded.", None
    
    if not question or question.strip() == "":
        return "Please enter a question.", None
    
    try:
        question = question.strip()
        log_message(f"Text processing: {question[:50]}...")
        
        # Process with return_audio=True for TTS
        result = state.orchestrator.process(question, return_audio=True)
        
        if result:
            response = result["response"]
            audio_path = result.get("audio_path", None)
            
            # Store in history
            state.history.append({"user": question, "assistant": response})
            state.conversation_id += 1
            
            log_message(f"Text response generated (Turn: {state.conversation_id})")
            log_message(f"   Response: {response[:100]}...")
            
            return response, audio_path
        else:
            return "❌ Query processing failed", None
            
    except Exception as e:
        log_message(f"❌ Error: {e}")
        import traceback
        log_message(traceback.format_exc())
        return f"❌ Error: {str(e)}", None


def process_audio_query(audio_file):
    """
    Process an audio query through ASR → RAG → TTS pipeline.
    
    Args:
        audio_file: Audio file from Gradio Audio component.
        
    Returns:
        tuple: (response_text, audio_output_path, history_html, conversation_id)
    """
    if audio_file is None:
        return "Please record or upload an audio file.", None, get_history_html(), state.conversation_id
    
    if not state.initialized or state.orchestrator is None:
        return "System not initialized.", None, get_history_html(), state.conversation_id
    
    if state.orchestrator.total_chunks_loaded == 0:
        return " No documents loaded.", None, get_history_html(), state.conversation_id
    
    try:
        # Get the audio file path
        audio_path = audio_file if isinstance(audio_file, str) else audio_file.name
        
        log_message(f"🎤 Audio processing: {audio_path}")
        
        # Process through orchestrator with return_audio=True
        result = state.orchestrator.process(audio_path, return_audio=True)
        
        if result:
            # Get the transcribed question
            question = result.get("question", "Unknown question")
            response = result["response"]
            audio_output = result.get("audio_path", None)
            
            # Store in history
            state.history.append({"user": f"[Audio] {question}", "assistant": response})
            state.conversation_id += 1
            
            log_message(f"Audio response generated (Turn: {state.conversation_id})")
            
            return response, audio_output, get_history_html(), state.conversation_id
        else:
            return "❌ Audio processing failed", None, get_history_html(), state.conversation_id
            
    except Exception as e:
        log_message(f"❌ Audio error: {e}")
        import traceback
        log_message(traceback.format_exc())
        return f"❌ Error: {str(e)}", None, get_history_html(), state.conversation_id


def reset_conversation():
    """
    Reset the conversation history and clear RAG memory.
    
    Returns:
        tuple: (status_message, history_html, conversation_id)
    """
    if state.orchestrator:
        state.orchestrator.clear_memory()
    state.history = []
    state.conversation_id = 0
    log_message("Conversation reset")
    return "Conversation reset!", get_history_html(), 0


def get_history_html():
    """
    Generate HTML representation of conversation history.
    
    Returns:
        str: HTML string for displaying conversation history in Gradio.
    """
    if not state.history:
        return """
        <div style="text-align: center; color: #888; padding: 40px;">
            <p>No conversation yet. Start by asking a question!</p>
        </div>
        """
    
    html = ""
    for i, turn in enumerate(state.history, 1):
        html += f"""
        <div style="margin-bottom: 15px;">
            <div style="background: #f0f7ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2196F3;">
                <strong style="color: #2196F3;">👤 User {i}:</strong> 
                <span style="color: #333;">{turn['user']}</span>
            </div>
            <div style="background: #f5f5f5; padding: 12px; border-radius: 8px; margin-top: 5px; border-left: 4px solid #4CAF50;">
                <strong style="color: #4CAF50;">🤖 Assistant:</strong>
                <div style="color: #333; white-space: pre-wrap;">{turn['assistant'].replace(chr(10), '<br>')}</div>
            </div>
        </div>
        <hr style="border: 0; border-top: 1px solid #e0e0e0; margin: 10px 0;">
        """
    return html


def upload_file(file):
    """
    Handle file upload and process document.
    
    Args:
        file: Uploaded file from Gradio File component.
        
    Returns:
        tuple: (status_message, total_chunks_display)
    """
    if file is None:
        return "No file uploaded", ""
    
    try:
        log_message(f"Uploading: {file.name}")
        
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())
        
        if state.orchestrator:
            state.orchestrator.add_uploaded_file(str(file_path))
            state.total_chunks = state.orchestrator.total_chunks_loaded
            state.source_count = len(state.orchestrator.source_history)
            log_message(f"Uploaded: {file.name} (Total: {state.total_chunks} chunks)")
            return f"Uploaded: {file.name}\nTotal chunks: {state.total_chunks}", f"Chunks: {state.total_chunks}"
        else:
            return "⚠️ Initialize system first", ""
            
    except Exception as e:
        log_message(f"❌ Error: {e}")
        return f"❌ Error: {str(e)}", ""


# ============================================================================
# UI
# ============================================================================

with gr.Blocks(title="🎙️ WE AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ WE AI Assistant
    Ask questions via **text** or **audio**. Get responses as **text** or **audio**.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Controls")
            init_btn = gr.Button("🚀 Initialize System", variant="primary")
            status = gr.Textbox(label="Status", value="Not initialized", interactive=False, lines=3)
            
            gr.Markdown("---")
            gr.Markdown("### 💬 Conversation")
            reset_btn = gr.Button("🔄 Reset Conversation", variant="secondary")
            reset_status = gr.Textbox(label="Reset Status", value="", interactive=False)
            turn_counter = gr.Number(label="Turns", value=0, interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### 📤 Upload Document")
            file_input = gr.File(label="Upload PDF, DOCX, TXT", file_types=[".pdf", ".docx", ".txt"])
            upload_btn = gr.Button("📄 Process Document")
            upload_status = gr.Textbox(label="Upload Status", value="", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### 📋 Logs")
            logs = gr.Textbox(label="Logs", lines=8, interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### 💡 Quick Questions")
            with gr.Row():
                q1 = gr.Button("140 سعر الخدمة")
                q2 = gr.Button("سعر الدقيقة")
            with gr.Row():
                q3 = gr.Button("باقات WE Space")
                q4 = gr.Button("خدمات WE")
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat History")
            history_display = gr.HTML(label="", value=get_history_html())
            
            gr.Markdown("---")
            
            # Tabs for Text and Audio input
            with gr.Tabs():
                with gr.TabItem("✏️ Text Input"):
                    gr.Markdown("Type your question below")
                    with gr.Row():
                        question = gr.Textbox(
                            label="", 
                            placeholder="Type your question here...", 
                            scale=4,
                            lines=2
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    answer = gr.Textbox(label="Answer", lines=4, interactive=False)
                    audio_output = gr.Audio(label="🔊 Audio Response", type="filepath")
                
                with gr.TabItem("🎤 Audio Input"):
                    gr.Markdown("Record or upload your question in audio")
                    audio_input = gr.Audio(
                        label="Record or Upload Audio",
                        type="filepath",
                        sources=["microphone", "upload"]
                    )
                    audio_submit = gr.Button("🎤 Process Audio", variant="primary")
                    audio_answer = gr.Textbox(label="Answer", lines=4, interactive=False)
                    audio_response_output = gr.Audio(label="🔊 Audio Response", type="filepath")
    
    # ========================================================================
    # Event Handlers
    # ========================================================================
    
    def on_init():
        """Handle initialization button click."""
        status_text = initialize_system()
        log_text = "\n".join(state.init_logs[-15:])
        history = get_history_html()
        return status_text, log_text, status_text, history, 0
    
    init_btn.click(
        on_init, 
        outputs=[status, logs, status, history_display, turn_counter]
    )
    
    # Text input handler
    def on_text_send(msg):
        """Handle text message submission."""
        if not msg or msg.strip() == "":
            return "", "Please enter a question.", None, get_history_html(), state.conversation_id
        
        response, audio = process_text_query(msg)
        history = get_history_html()
        return "", response, audio, history, state.conversation_id
    
    send_btn.click(
        on_text_send, 
        inputs=[question], 
        outputs=[question, answer, audio_output, history_display, turn_counter]
    )
    question.submit(
        on_text_send, 
        inputs=[question], 
        outputs=[question, answer, audio_output, history_display, turn_counter]
    )
    
    # Audio input handler
    def on_audio_send(audio):
        """Handle audio submission."""
        if audio is None:
            return "Please record or upload audio.", None, get_history_html(), state.conversation_id
        
        response, audio_out, history, turns = process_audio_query(audio)
        return response, audio_out, history, turns
    
    audio_submit.click(
        on_audio_send,
        inputs=[audio_input],
        outputs=[audio_answer, audio_response_output, history_display, turn_counter]
    )
    
    # Quick questions (text)
    def on_quick(q):
        """Handle quick question button clicks."""
        response, audio = process_text_query(q)
        history = get_history_html()
        return q, response, audio, history, state.conversation_id
    
    q1.click(lambda: on_quick("ما هو سعر خدمة 140 دليل؟"), outputs=[question, answer, audio_output, history_display, turn_counter])
    q2.click(lambda: on_quick("كم سعر الدقيقة؟"), outputs=[question, answer, audio_output, history_display, turn_counter])
    q3.click(lambda: on_quick("ما هي باقات WE Space؟"), outputs=[question, answer, audio_output, history_display, turn_counter])
    q4.click(lambda: on_quick("ما هي خدمات WE؟"), outputs=[question, answer, audio_output, history_display, turn_counter])
    
    # Reset conversation
    def on_reset():
        """Handle conversation reset."""
        status_msg, history, turns = reset_conversation()
        return status_msg, history, turns
    
    reset_btn.click(
        on_reset,
        outputs=[reset_status, history_display, turn_counter]
    )
    
    # Upload file
    def on_upload(file):
        """Handle document upload."""
        status_text, stats = upload_file(file)
        log_text = "\n".join(state.init_logs[-8:])
        return status_text, stats, log_text
    
    upload_btn.click(
        on_upload, 
        inputs=[file_input], 
        outputs=[upload_status, status, logs]
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=False)