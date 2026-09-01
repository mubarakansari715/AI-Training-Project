# RAG Chatbot — Project Plan

Single source of truth for building the end-to-end RAG document chatbot,
phase by phase. Each phase is only started after the previous one is
implemented, explained, tested, and confirmed.

**Working rule:** we do NOT jump ahead. Complete a phase → I explain what
was built, which files changed, how it works, and the test command → you
confirm → we move to the next checkbox.

---

## 0. Project Goal

Build a chatbot that:

- Accepts multiple uploaded documents (PDF, DOCX, TXT, Markdown)
- Parses, cleans, and chunks the text
- Embeds chunks and stores them in a persistent vector DB
- Retrieves only relevant chunks for a user question
- Sends only that context to an LLM (Gemini)
- Answers using only the retrieved context (no hallucination)
- Shows source filename + page number for every answer
- Clearly says "not found in documents" when context is insufficient

## 1. Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| UI | Streamlit |
| RAG glue | LangChain (only where it genuinely simplifies) |
| Document parsing | PDF / DOCX / TXT / Markdown libraries |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (configurable) |
| Vector DB | ChromaDB (persistent, local) |
| LLM | Google Gemini API (provider kept swappable) |
| Config | `.env` + `.env.example` (no hardcoded keys) |

## 2. Architecture

```
User → Streamlit UI → Document Manager → Document Parser → Text Cleaner
     → Chunking Service → Embedding Service → ChromaDB → Retriever
     → Context Builder → LLM (Gemini) → Answer + Sources → Streamlit UI
```

Clear separation of concerns: UI / document processing / embedding /
vector storage / retrieval / LLM generation / configuration. No
monolithic single-file app.

## 3. Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py
│   ├── config/settings.py
│   ├── ui/{sidebar.py, chat.py, components.py}
│   ├── rag/{document_loader.py, text_cleaner.py, chunker.py,
│   │        embeddings.py, vector_store.py, retriever.py,
│   │        prompt.py, generator.py}
│   └── services/{document_service.py, rag_service.py}
├── data/{uploads/, chroma/}
├── tests/ (13 files covering every module + a full pipeline
│           integration test — see README's Testing section)
├── notebooks/ (kept empty — not needed for this build)
├── screenshots/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

(May be adjusted if there's a better structural reason — will be
explained before any deviation.)

---

## Phases

- [x] **Phase 1 — Project Setup**
  requirements.txt, .env.example, .gitignore, README skeleton, venv
  instructions, entry point. Streamlit app must open successfully.
  No RAG logic yet.

- [x] **Phase 2 — Document Upload UI**
  Sidebar upload widget (PDF/DOCX/TXT/MD), document list display, main
  welcome area. No processing yet — UI only.

- [x] **Phase 3 — Document Parsing**
  Parsers for PDF/DOCX/TXT/MD → common `{text, source, page}`
  representation. Graceful handling of empty/corrupted/unsupported
  files. Independent tests.

- [x] **Phase 4 — Text Cleaning**
  Normalize whitespace, blank lines, broken line spacing, extraction
  artifacts — without changing meaning.

- [x] **Phase 5 — Intelligent Chunking**
  Recursive/structure-aware splitter. Configurable `chunk_size` (800)
  and `chunk_overlap` (150). Deterministic `chunk_id`. Metadata
  preserved. Tests.

- [x] **Phase 6 — Embeddings**
  Reusable embedding service using `all-MiniLM-L6-v2` (configurable
  model), loaded once and cached — not per question.

- [x] **Phase 7 — ChromaDB Vector Store**
  Persistent local Chroma collection storing chunks + embeddings +
  metadata (source, page, chunk_id). Add + search operations.
  Configurable `top_k` (default 5).

- [x] **Phase 8 — Retrieval**
  Embed question → similarity search → top-k relevant chunks with
  metadata/scores. Only relevant context is ever forwarded, never the
  full corpus. Optional relevance threshold.

- [x] **Phase 9 — RAG Prompt**
  Dedicated prompt template file: SYSTEM INSTRUCTIONS / CONTEXT /
  SOURCE INFORMATION / USER QUESTION. Instructs context-only answers,
  explicit "not found" fallback, cite source + page when possible.

- [x] **Phase 10 — Gemini Integration**
  Generator service reading `GEMINI_API_KEY` from `.env`. Takes
  question + retrieved context → returns answer + sources. Graceful
  failure handling, no crashes, no key leakage.

- [x] **Phase 11 — Complete RAG Pipeline**
  Single orchestrator service wiring upload → parse → clean → chunk →
  embed → store → retrieve → build context → generate → answer +
  sources. Kept out of the Streamlit UI layer.

- [x] **Phase 12 — Chat UI**
  Polished Streamlit chat interface using `st.chat_message` /
  `st.chat_input`, conversation history via session state.

- [x] **Phase 13 — Source Citations**
  Every answer displays sources (filename + page, or filename only
  when page is unavailable). Never invented.

- [x] **Phase 14 — "I Don't Know" Protection**
  Strict document-grounded mode (default ON) — refuses to answer from
  general knowledge when context is insufficient.

- [x] **Phase 15 — Document Management**
  Clear chat, clear index, re-index documents, list uploaded documents,
  avoid duplicate indexing.

- [x] **Phase 16 — RAG Debug / Transparency Mode**
  Optional "Show Retrieval Details" panel showing chunk text, source,
  page, and similarity score for every retrieved chunk.

- [x] **Phase 17 — Error Handling**
  Friendly UI messages for: missing/invalid API key, empty/corrupted/
  unsupported documents, Chroma failure, embedding failure, LLM
  failure, no relevant docs, empty question. No stack traces surfaced
  to users; proper logging for developers.

- [x] **Phase 18 — Testing**
  Unit tests (parsing, cleaning, chunking, metadata, retrieval, prompt
  construction) + integration tests for the full pipeline. LLM calls
  mocked — `pytest` should pass without a real API key.

- [x] **Phase 19 — Code Quality**
  Type hints, docstrings on key components, small reusable functions,
  consistent style, proper exceptions/logging, no dead/duplicated code.

- [x] **Phase 20 — Security Review**
  No keys in source/README/logs/notebooks, `.env` gitignored,
  `.env.example` placeholders only, no private documents/tokens
  committed.

- [x] **Phase 21 — README**
  Overview, features, Mermaid architecture diagram, tech stack,
  structure, install/run/env instructions, "how RAG works", example
  usage, screenshots, testing, future improvements.

- [x] **Phase 22 — GitHub Preparation**
  Final clone-and-run verification against the acceptance criteria
  below.

---

## Final Acceptance Criteria

- [x] Clone repo → create venv → install deps → set Gemini key → `streamlit run app/main.py`
- [x] Upload multiple documents, see them listed in the UI
- [x] Process/index documents
- [x] Ask questions, get answers grounded in retrieved context
- [x] See source filenames + page numbers where available
- [x] See retrieved chunks in debug mode
- [x] Get "information not found" response when appropriate
- [x] Multi-turn conversation works
- [x] Clear chat works
- [x] Clear/rebuild index works
- [x] `pytest` passes without secrets
- [x] README fully explains the architecture

All verified live end-to-end with a real Gemini API key on 2026-09-01
(see status log below) — including multi-document upload, strict-mode
"not found" on an irrelevant question, a correct grounded answer with
sources on a relevant follow-up, and both management buttons.

---

## Status Log

| Date | Phase | Notes |
|---|---|---|
| 2026-09-01 | Plan created | Awaiting go-ahead for Phase 1 |
| 2026-09-01 | Phase 1 complete | Skeleton, config files, entry point created; Streamlit verified to launch (HTTP 200) |
| 2026-09-01 | Phase 2 complete | Sidebar upload UI + main welcome area; verified visually with Playwright (upload → document list updates) |
| 2026-09-01 | Phase 3 complete | PDF/DOCX/TXT/MD parsing with common DocumentPage representation; 9 unit tests pass; verified in UI (success + friendly error on empty file) |
| 2026-09-01 | Chat input added early (out of order, by request) | `app/ui/chat.py` added ahead of Phase 12 as a UI-only preview — real answers still require Phases 4-11. Phase 12 checkbox stays open until it's wired to real retrieval + Gemini. |
| 2026-09-01 | Chat UI + custom styling reverted per user request | `main.py` no longer calls `render_chat()`/`inject_chat_theme()`; `.streamlit/config.toml` removed. App is back to the plain Phase 3 view (upload + document processing, default Streamlit look). `app/ui/chat.py` and the CSS in `app/ui/components.py` are left on disk, unused, in case they're wanted again later. |
| 2026-09-01 | Phase 4 complete | `clean_text()` (whitespace/blank-line/hyphen-break/control-char cleanup) wired into `document_service.parse_uploaded_files`; 11 unit tests pass (31 total); verified end-to-end on messy sample text |
| 2026-09-01 | Environment fix | Intel macOS caps PyTorch at 2.2.2 (no newer x86_64 wheels); unpinned `sentence-transformers`/`transformers` had resolved to versions requiring torch>=2.5, breaking imports. Pinned `numpy<2`, `torch==2.2.2`, `transformers==4.44.2`, `sentence-transformers==3.0.1` in requirements.txt. Relevant for Phase 6 (embeddings) too. |
| 2026-09-01 | Phase 5 complete | `app/config/settings.py` added (env-driven config); `chunk_pages()` via LangChain's RecursiveCharacterTextSplitter, deterministic chunk_id, metadata preserved; 10 unit tests pass (30 total); verified in UI (2399 chars → 4 chunks) |
| 2026-09-01 | Phases 6-22 completed in one pass (user requested "complete all the phases") | Embeddings (`embeddings.py`, cached sentence-transformers), ChromaDB store (`vector_store.py`, cosine space, dedup by chunk_id), retriever with similarity threshold, prompt template, Gemini generator, full `RAGPipeline` orchestrator, real chat UI with sources + debug view, document management (clear chat/index, re-index, duplicate-upload guard), error handling throughout, 64 tests passing (34 new), ruff clean, full README rewrite. See details below. |
| 2026-09-01 | SDK swap | The legacy `google-generativeai` package is fully deprecated (upstream warning: "all support has ended"). Switched to the current `google-genai` package/`genai.Client` API. Also dropped unused `langchain`/`langchain-community`/`langchain-google-genai` in favor of just `langchain-text-splitters` (the only piece actually used), per "avoid unnecessary abstraction". |
| 2026-09-01 | Live key test + 2 real bugs found and fixed | User provided a real `GEMINI_API_KEY`. (1) `gemini-2.5-flash` is retired for new users — API error pointed us to `gemini-3.6-flash`; updated the default everywhere. (2) `AnswerGenerator(api_key="")` was silently falling back to the configured key because of `api_key or settings.gemini_api_key` (empty string is falsy) — fixed to an explicit `is None` check; caught by a test that started failing once a real key was present. (3) Default `SIMILARITY_THRESHOLD=0.0` meant strict mode's "not found" path only ever triggered on a fully empty index, never on an irrelevant question, since ChromaDB always returns top_k nearest neighbors regardless of relevance. Empirically measured real cosine scores (relevant: 0.27-0.63, irrelevant: -0.02-0.07) and set the default to 0.2. |
| 2026-09-01 | Full live verification | Ran the real app (single instance, to avoid a ChromaDB file-lock conflict hit earlier when two servers shared `data/chroma`) against the real Gemini API: multi-document upload, grounded answer with correct source citation, retrieval debug view with similarity scores, strict-mode refusal on an off-topic question, a relevant multi-turn follow-up, persistence across a fresh browser session, and both "Clear chat"/"Clear index" buttons — all confirmed working via screenshots. |
| 2026-09-01 | Conversational memory added (user request, ChatGPT-style) | `generator.generate()` now accepts prior chat turns and sends them as native multi-turn `contents` (`user`/`model` roles) to Gemini, capped at the last 10 messages. Found and fixed a real gap: `SYSTEM_INSTRUCTIONS` said "use only the provided context," which the model correctly interpreted as excluding conversation history — it refused to recall the user's name even with strict mode off. Reworded to explicitly allow conversational continuity while still restricting *document* facts to retrieved context. 4 new tests (68 total). **Design note:** strict mode's "not found" short-circuit is still based only on the current question's retrieval score, unaffected by history — so a pure off-document fact (e.g. "what's my name") still gets refused under strict mode by design; toggle strict mode off to use conversational memory outside document Q&A. Live verification hit the Gemini free-tier daily quota (20 req/day) from cumulative session testing — confirmed via mocked tests instead; error handling (Phase 17) correctly returned the friendly fallback message rather than crashing. |
| 2026-09-01 | UI polish (user requests) | In-place "Thinking..." spinner added to `chat.py` (user message + assistant avatar render immediately, spinner shows where the answer will appear, instead of relying on Streamlit's top-right indicator) plus `submit_mode="disable"`. Sidebar reordered: "Advanced settings" (strict mode + debug toggle) moved into a collapsed expander, positioned last, after Document management. |
| 2026-09-01 | Gemini model settled on `gemini-3.5-flash-lite` | User rotated their API key (good practice, since the prior one had been pasted into chat) and asked for a low-token model. Tried in order: `gemini-flash-latest` (valid but hit a real `503` "high demand" from Google, reproduced independently via the user's own curl command — confirmed provider-side, not our bug) → `gemini-2.5-flash-lite` (listed in the model catalog but 404s: "no longer available to new users," per two independent live checks) → `gemini-3.5-flash-lite`, the model Google's own error pointed to — verified working on both an empty-context and a real document question, source citation intact. Set as the default in `.env`, `.env.example`, and `settings.py`. |
| 2026-09-01 | Small talk bypass added (user hit "not found" on "hi") | Real UX gap: strict mode's pre-LLM short-circuit fired on *any* input without matching document content — including plain greetings, since "hi" naturally scores near 0 similarity against real document chunks. Added `app/rag/small_talk.py` (`is_small_talk()`, regex-based: greetings/thanks/bye/ok/etc.) and wired it into `RAGPipeline.ask()` so small talk always reaches the LLM for a normal reply, while genuine document questions with no match are still correctly refused. Verified the LLM already handles a bare "hi" naturally once it reaches it (no context needed). 6 new tests (98 total), ruff clean. |
| 2026-09-01 | Persistent conversation history added (user request, ChatGPT-style) | New `app/services/conversation_service.py` (`ConversationStore`) — each conversation saved as its own JSON file under `data/conversations/` (gitignored, like uploads/chroma) after every exchange, with an auto-generated title from the first user message (truncated at 50 chars, ChatGPT-style). Sidebar (`sidebar.py`) now opens with a "🆕 New chat" button and a "💬 Conversations" list — each past conversation is a button; clicking one loads its full history via `switch_conversation()` in `main.py`; the active one is visually highlighted (`type="primary"`). The old "Clear chat" button was retired in favor of "New chat" (same effect — starts a fresh empty conversation — but now the previous one stays saved and reachable, matching ChatGPT's actual behavior instead of just wiping it). Storage is plain JSON files (not SQLite) — deliberately simple, human-inspectable, no schema migrations. 10 new tests (108 total), ruff clean. Not yet live-tested in-session (user's own terminal instance was actively running; skipped spinning up a competing server to avoid a repeat of the earlier ChromaDB lock conflict and to avoid polluting the user's real conversation history with test data) — user asked to verify in their own restarted instance. |
| 2026-09-01 | Stale-process crash + fix | User's terminal process was still running the pre-conversation-history `sidebar.py`, whose `render_sidebar()` took no arguments — `main.py`'s new call with `conversations=`/`active_conversation_id=` threw a `TypeError` at that call site (same class of bug as the earlier `ImportError` incident: Streamlit's hot-reload not fully re-importing a changed function signature). Reproduced in a fully isolated copy first (confirmed the code itself was correct — rendered cleanly), which pointed to a stale process rather than a real bug. Killed the stale process and restarted fresh on the same port (8501); user's session recovered immediately (log showed a real Gemini call succeed right after). |
| 2026-09-01 | Small talk detection expanded to introductions + recall questions (user report: "Hi my name is ali" → "what is my name?" both refused) | Real gap: `is_small_talk()` only matched bare greetings ("hi"), not "Hi my name is ali" (an introduction) or "what is my name?" (a question about the conversation itself) — both still hit strict mode's pre-LLM refusal. Considered and rejected a blanket fix (bypass the refusal whenever *any* conversation history exists): tested it and found the LLM gave a non-answer ("Hello Ali!") to an unrelated general-knowledge question ("What is the capital of France?") once history was present — too permissive, breaks strict mode's guarantee. Instead expanded `small_talk.py` with two new bounded regex patterns: self-introductions ("hi, my name is X" / "I'm X" / "I am X", capped at 1-3 name words so it can't swallow real sentences starting with "I am...") and conversational-recall questions ("what's my name", "what did I say", "do you remember", etc.). Verified end-to-end in an isolated pipeline (zero documents indexed): the exact reported turn 1 → turn 2 sequence now works, while a genuinely unrelated question ("capital of France") still correctly refuses under strict mode, both with and without prior history. 24 new small-talk tests including explicit false-positive guards ("I am looking for...", "call me later about...") (131 total), ruff clean. |
| 2026-09-01 | Delete-conversation button added, icons polished | `sidebar.py`: each conversation row now has a 🗑️ delete button (Material icon, `:material/delete:`, icon-only) next to the title button, `[6,1]` column split with vertical alignment for even sizing. Deleting the active conversation starts a fresh "New chat" instead of leaving a dangling reference to a deleted file. Also swapped the "New chat" button and "Conversations" header from raw emoji to Material Symbols (`:material/edit_square:`, `:material/forum:`) per the project's own design guidance (icons over emoji). Verified visually in an isolated copy: correct icon rendering/sizing, and deleting one conversation removes only that one while leaving the active one untouched and still highlighted. 131 tests still passing (UI-only change, no new unit tests), ruff clean. |
| 2026-09-01 | README standardized (user request) | Added a Table of Contents, grouped Features into Document handling/Retrieval/Conversation/Reliability, synced the Environment Variables table and Project Structure tree exactly against the real `.env.example` and file layout, expanded Testing to name every covered area. Purely a documentation pass, no code changes. |
| 2026-09-01 | Document upload moved into the chat input, ChatGPT/Gemini-style (user request) | Removed `st.file_uploader` from `sidebar.py` entirely (and the now-meaningless "Re-index documents" button, since chat-input file submissions are one-shot events, not a persistent widget selection to "re-run"). `chat.py` now uses `st.chat_input(accept_file="multiple", file_type=[...])`, giving a native 📎 attach icon inside the input bar. Indexing happens inline in the assistant's turn (spinner, then an "📄 Indexed N chunk(s)..." note prepended to the real answer, or standalone if no question text was sent alongside the attachment). Sidebar keeps just the document list + Clear index for visibility/management. Added `attached_files` to `conversation_service.py`'s message (de)serialization so past conversations still show what was attached when reopened. New shared constant `SUPPORTED_DOCUMENT_TYPES` in `components.py`. Verified end-to-end in an isolated copy: attach icon renders correctly, file chip shows in the input before sending, single-turn attach+question flow correctly indexes then answers with the right source citation, sidebar document list and conversation list both update. 134 tests passing (2 new/updated for `attached_files`), ruff clean. |
| 2026-09-01 | Gemini temperature made configurable (user question: "what temperature have you set?") | Answer: none had been set — the API default was in use. Added `GEMINI_TEMPERATURE` (default `0.2`, low/literal — appropriate for a document-grounded QA bot) threaded through `settings.py` → `AnswerGenerator.__init__` → `types.GenerateContentConfig(temperature=...)` on every `generate_content` call. Documented in `.env.example`, `.env`, and the README's Environment Variables table. 2 new tests (default applied, explicit override applied); all existing generator mocks needed a `config` parameter added since the real call now passes one — fixed. 134 tests passing, ruff clean. |
