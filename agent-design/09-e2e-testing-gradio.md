# End-to-End Testing & Gradio UI — Verification and Polish

> Deep dive: Day 7 | Topic: E2E test scenarios, Gradio HITL integration, streaming

---

## Q1. How do you write deterministic end-to-end tests for an agent with interrupts?

**Answer:**

Agent tests are tricky because: (1) LLM outputs are non-deterministic, (2) `interrupt()` pauses execution and needs a resume call, (3) tool results may vary (real web search vs mock).

**Strategy: Mock the LLM and tools; test the graph wiring.**

```python
import pytest
from unittest.mock import MagicMock, patch
from langgraph.types import Command

# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """LLM that returns deterministic tool calls or text responses."""
    llm = MagicMock()
    return llm

@pytest.fixture
def agent_graph(mock_llm):
    """Build the real graph but inject mock LLM and tools."""
    from agent import build_graph
    return build_graph(llm=mock_llm, use_mock_tools=True)

# ─── Test 1: Web search routing ──────────────────────────────────────────────

def test_web_search_no_hitl(agent_graph, mock_llm):
    """Web search should route directly to tool execution — no HITL fires."""
    # Configure LLM to select web_search tool
    mock_llm.invoke.side_effect = [
        make_tool_call_response("web_search", {"query": "LangGraph release date"}),
        make_text_response("LangGraph was released in January 2024."),
    ]

    config = {"configurable": {"thread_id": "test-web-1"}}
    result = agent_graph.invoke(
        {"messages": [{"role": "user", "content": "When was LangGraph released?"}]},
        config=config,
    )

    # No interrupt should have fired
    assert "__interrupt__" not in result
    
    # Tool execution should have happened
    state = agent_graph.get_state(config)
    assert state.values.get("last_tool_used") == "web_search"
    assert state.values.get("approval_status") is None  # HITL never triggered

# ─── Test 2: Note saving fires HITL ─────────────────────────────────────────

def test_note_saving_triggers_hitl(agent_graph, mock_llm):
    """Note saving must pause at the approval gate."""
    mock_llm.invoke.return_value = make_tool_call_response(
        "save_note",
        {"title": "LangGraph", "content": "Graph-based agent framework by LangChain"}
    )

    config = {"configurable": {"thread_id": "test-hitl-1"}}
    result = agent_graph.invoke(
        {"messages": [{"role": "user", "content": "Save a note about LangGraph"}]},
        config=config,
    )

    # Execution should have paused
    assert "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["type"] == "approval_request"
    assert interrupt_payload["tool"] == "save_note"
    assert "Graph-based agent framework" in interrupt_payload["content"]

def test_note_saving_approve_persists(agent_graph, mock_llm, tmp_path):
    """Approving a note should result in it being saved."""
    mock_llm.invoke.side_effect = [
        make_tool_call_response("save_note", {
            "title": "LangGraph", "content": "Graph-based framework"
        }),
        make_text_response("Note saved successfully."),
    ]

    config = {"configurable": {"thread_id": "test-hitl-2"}}
    
    # First call — should pause at approval gate
    agent_graph.invoke(
        {"messages": [{"role": "user", "content": "Save a note about LangGraph"}]},
        config=config,
    )

    # Resume with approval
    result = agent_graph.invoke(Command(resume={"action": "approve"}), config=config)

    # Verify note was actually saved
    import json
    notes_file = tmp_path / "notes.json"
    if notes_file.exists():
        notes = json.loads(notes_file.read_text())
        assert any("LangGraph" in n.get("title", "") for n in notes)
    
    state = agent_graph.get_state(config)
    assert state.values.get("approval_status") == "approved"

def test_note_saving_reject_does_not_persist(agent_graph, mock_llm):
    """Rejecting should clear pending_tool and skip tool execution."""
    mock_llm.invoke.return_value = make_tool_call_response("save_note", {
        "title": "Test", "content": "Should not be saved"
    })

    config = {"configurable": {"thread_id": "test-hitl-3"}}
    agent_graph.invoke(
        {"messages": [{"role": "user", "content": "Save this note"}]},
        config=config,
    )

    result = agent_graph.invoke(Command(resume={"action": "reject"}), config=config)
    
    state = agent_graph.get_state(config)
    assert state.values.get("approval_status") == "rejected"
    assert state.values.get("pending_tool") is None

# ─── Test 3: Memory recall ───────────────────────────────────────────────────

def test_memory_recall_from_earlier_turn(agent_graph, mock_llm):
    """Agent should retrieve facts from ChromaDB saved in earlier turns."""
    config = {"configurable": {"thread_id": "test-memory-1"}}
    
    # Simulate that ChromaDB already has a relevant fact
    with patch("agent.memory.query_facts") as mock_query:
        mock_query.return_value = ["LangGraph: Graph-based framework, released Jan 2024"]
        mock_llm.invoke.return_value = make_text_response(
            "Based on my notes, LangGraph is a graph-based framework released in January 2024."
        )
        
        result = agent_graph.invoke(
            {"messages": [{"role": "user", "content": "What do you know about LangGraph?"}]},
            config=config,
        )
    
    # Verify memory was queried
    mock_query.assert_called_once()
    
    # Response should contain the recalled fact
    final_msg = [m for m in result["messages"] if m.get("role") == "assistant"][-1]
    assert "graph-based" in final_msg["content"].lower()

# ─── Test 4: Summarization triggers ─────────────────────────────────────────

def test_summarization_triggers_at_2000_tokens(agent_graph, mock_llm):
    """Summarization must kick in when token count exceeds 2000."""
    # Pre-load state with lots of messages
    long_messages = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": "This is a filler message about AI systems and agent design " * 5}
        for i in range(45)  # ~2250 tokens
    ]
    
    config = {"configurable": {"thread_id": "test-summary-1"}}
    
    with patch("agent.maybe_summarize") as mock_summarize:
        mock_summarize.return_value = {
            "messages": [{"role": "system", "content": "[Conversation summary]..."}] + long_messages[-4:],
            "summary": "User discussed AI systems at length.",
            "token_count": 450,
        }
        mock_llm.invoke.return_value = make_text_response("Got it.")
        
        result = agent_graph.invoke(
            {"messages": long_messages + [{"role": "user", "content": "Continue"}]},
            config=config,
        )
    
    mock_summarize.assert_called_once()

# ─── Test 5: Feedback collection ────────────────────────────────────────────

def test_feedback_stored_in_sqlite(agent_graph, mock_llm, tmp_path):
    """Thumbs down rating must be persisted to SQLite."""
    import sqlite3
    
    mock_llm.invoke.return_value = make_text_response("The weather in 2035 is unknown.")
    config = {"configurable": {"thread_id": "test-feedback-1"}}
    
    # First call — agent responds, then pauses for feedback
    agent_graph.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in 2035?"}]},
        config=config,
    )
    
    # Resume with thumbs down
    agent_graph.invoke(Command(resume={"action": "thumbs_down"}), config=config)
    
    # Check SQLite
    conn = sqlite3.connect("agent_feedback.db")
    rows = conn.execute(
        "SELECT rating FROM feedback WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        ("test-feedback-1",)
    ).fetchall()
    conn.close()
    
    assert len(rows) > 0
    assert rows[0][0] == -1  # thumbs down = -1

# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_tool_call_response(tool_name: str, args: dict):
    """Create a mock LLM response that calls a tool."""
    response = MagicMock()
    response.content = ""
    response.tool_calls = [{"name": tool_name, "args": args, "id": "tc-mock-001"}]
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return response

def make_text_response(text: str):
    """Create a mock LLM response with text content."""
    response = MagicMock()
    response.content = text
    response.tool_calls = []
    response.usage = MagicMock(prompt_tokens=80, completion_tokens=40)
    return response
```

---

## Q2. How do you build the Gradio interface with dynamic HITL approve/reject buttons?

**Answer:**

The key challenge: Gradio is request-response by default, but HITL creates a two-phase interaction (submit → approve/reject). Solution: track pending HITL state in a `gr.State` object.

```python
import gradio as gr
from langgraph.types import Command
from agent import build_graph

graph = build_graph()

def handle_message(user_msg: str, history: list, app_state: dict):
    """Process a user message. Returns updated history + HITL panel visibility."""
    thread_id = app_state.get("thread_id", "gradio-session-1")
    config    = {"configurable": {"thread_id": thread_id}}
    turn      = app_state.get("turn", 0) + 1

    response_text = "..."
    hitl_active   = False
    hitl_preview  = ""
    memory_text   = app_state.get("memory_text", "_No facts yet_")

    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=config,
        )

        if "__interrupt__" in result:
            ivs = result["__interrupt__"]
            payload = ivs[0].value if ivs else {}

            if payload.get("type") == "approval_request":
                hitl_active  = True
                hitl_preview = (
                    f"**📝 About to save note:**\n\n"
                    f"**Title:** {payload.get('title', 'untitled')}\n\n"
                    f"**Content:**\n{payload.get('content', '')}\n\n"
                    f"*Approve, reject, or edit before saving.*"
                )
                response_text = "⏸️ Waiting for your approval before saving this note..."

            elif payload.get("type") == "feedback_request":
                # Non-blocking feedback — just store that we need it
                app_state["pending_feedback"] = True
                # Extract final response from state
                state = graph.get_state(config)
                msgs = state.values.get("messages", [])
                ai_msgs = [m for m in msgs if getattr(m, "type", None) == "ai" or
                           (isinstance(m, dict) and m.get("role") == "assistant")]
                if ai_msgs:
                    response_text = getattr(ai_msgs[-1], "content", ai_msgs[-1].get("content", ""))
        else:
            # Normal response — extract assistant message
            msgs = result.get("messages", [])
            ai_msgs = [m for m in msgs
                       if (hasattr(m, "type") and m.type == "ai") or
                          (isinstance(m, dict) and m.get("role") == "assistant")]
            if ai_msgs:
                last = ai_msgs[-1]
                response_text = getattr(last, "content", last.get("content", ""))

            # Update memory sidebar
            state = graph.get_state(config)
            retrieved = state.values.get("last_retrieved_facts", [])
            if retrieved:
                memory_text = "**Retrieved facts:**\n" + "\n".join(f"• {f}" for f in retrieved)

    except Exception as e:
        response_text = f"❌ Error: {str(e)}"

    # Update state
    new_app_state = {
        **app_state,
        "thread_id":   thread_id,
        "turn":        turn,
        "hitl_active": hitl_active,
        "memory_text": memory_text,
    }

    history = history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": response_text},
    ]

    return (
        history,
        gr.update(visible=hitl_active),
        hitl_preview,
        memory_text,
        new_app_state,
        ""  # Clear input box
    )

def handle_approve(app_state: dict, history: list):
    thread_id = app_state.get("thread_id", "gradio-session-1")
    config    = {"configurable": {"thread_id": thread_id}}

    graph.invoke(Command(resume={"action": "approve"}), config=config)
    
    # Get final state
    state = graph.get_state(config)
    msgs  = state.values.get("messages", [])
    ai_msgs = [m for m in msgs if hasattr(m, "type") and m.type == "ai"]
    final_text = ai_msgs[-1].content if ai_msgs else "✅ Note saved."

    history = history + [{"role": "assistant", "content": f"✅ Note approved and saved.\n\n{final_text}"}]

    return history, gr.update(visible=False), "", {**app_state, "hitl_active": False}

def handle_reject(app_state: dict, history: list):
    thread_id = app_state.get("thread_id", "gradio-session-1")
    config    = {"configurable": {"thread_id": thread_id}}

    graph.invoke(Command(resume={"action": "reject"}), config=config)
    history = history + [{"role": "assistant", "content": "❌ Note rejected. Nothing was saved."}]

    return history, gr.update(visible=False), "", {**app_state, "hitl_active": False}

def handle_thumbs(action: str, app_state: dict):
    thread_id = app_state.get("thread_id", "gradio-session-1")
    config    = {"configurable": {"thread_id": thread_id}}

    if app_state.get("pending_feedback"):
        graph.invoke(Command(resume={"action": action}), config=config)
        new_state = {**app_state, "pending_feedback": False}
        label = "👍 Thanks!" if action == "thumbs_up" else "👎 Noted!"
        return new_state, label
    return app_state, ""

# ─── Build UI ─────────────────────────────────────────────────────────────

with gr.Blocks(
    title="🤖 Personal Research Assistant",
    theme=gr.themes.Soft(),
    css=".hitl-panel { border: 2px solid #f59e0b; border-radius: 8px; padding: 12px; }"
) as demo:

    gr.Markdown("# 🤖 Personal Research Assistant\n*Memory • Web Search • Notes • Calendar*")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot   = gr.Chatbot(type="messages", height=480, show_label=False)
            msg_input = gr.Textbox(
                placeholder="Ask anything — I remember our conversation...",
                label="Message", lines=2
            )
            with gr.Row():
                send_btn   = gr.Button("Send ↵", variant="primary", scale=3)
                thumbs_up  = gr.Button("👍", scale=1)
                thumbs_dn  = gr.Button("👎", scale=1)
            feedback_label = gr.Markdown("")

        with gr.Column(scale=1):
            gr.Markdown("### 📚 Retrieved Memory")
            memory_display = gr.Markdown("_No facts retrieved yet_")
            gr.Markdown("---")
            gr.Markdown("### 📊 Session Info")
            session_info = gr.Markdown("Turn: 0")

    # HITL approval panel — hidden by default
    with gr.Group(visible=False, elem_classes="hitl-panel") as hitl_panel:
        gr.Markdown("### ⚠️ Approval Required")
        hitl_preview = gr.Markdown("")
        with gr.Row():
            approve_btn = gr.Button("✅ Approve & Save", variant="primary")
            reject_btn  = gr.Button("❌ Reject",         variant="stop")

    # Session state
    app_state = gr.State({"thread_id": "gradio-1", "turn": 0, "hitl_active": False})

    # Wire events
    send_btn.click(
        handle_message,
        inputs=[msg_input, chatbot, app_state],
        outputs=[chatbot, hitl_panel, hitl_preview, memory_display, app_state, msg_input],
    )
    approve_btn.click(handle_approve, [app_state, chatbot], [chatbot, hitl_panel, hitl_preview, app_state])
    reject_btn.click( handle_reject,  [app_state, chatbot], [chatbot, hitl_panel, hitl_preview, app_state])
    thumbs_up.click(handle_thumbs, inputs=[gr.State("thumbs_up"), app_state], outputs=[app_state, feedback_label])
    thumbs_dn.click(handle_thumbs, inputs=[gr.State("thumbs_down"), app_state], outputs=[app_state, feedback_label])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

---

## Q3. How do you add streaming to the Gradio interface?

**Answer:**

LangGraph streaming with `stream_mode="messages"` yields `(AIMessageChunk, metadata)` tuples:

```python
def streaming_chat(user_msg: str, history: list, app_state: dict):
    """Generator function — yields updated history after each token."""
    thread_id = app_state.get("thread_id", "gradio-1")
    config    = {"configurable": {"thread_id": thread_id}}

    # Append user message and empty AI placeholder
    history = history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": ""},
    ]
    yield history, gr.update(visible=False), "", app_state, ""

    full_response = ""
    hitl_fired    = False

    for chunk, metadata in graph.stream(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
        stream_mode="messages",
    ):
        # Check for HITL interrupt in metadata
        if metadata.get("langgraph_step") and "__interrupt__" in str(chunk):
            hitl_fired = True
            history[-1]["content"] = "⏸️ Waiting for approval..."
            yield history, gr.update(visible=True), "Approval needed...", app_state, ""
            return

        # Accumulate streaming tokens
        if hasattr(chunk, "content") and isinstance(chunk.content, str):
            full_response += chunk.content
            history[-1]["content"] = full_response
            yield history, gr.update(visible=False), "", app_state, ""

# In Gradio — use .then() for streaming:
send_btn.click(
    streaming_chat,
    inputs=[msg_input, chatbot, app_state],
    outputs=[chatbot, hitl_panel, hitl_preview, app_state, msg_input],
)
```

**Critical note:** Gradio requires the function to be a `generator` (use `yield` not `return`) for streaming to work. Every `yield` updates the UI.

---

## Evaluation Checklist (Day 7 Deliverable)

```
□ Turn 1: Web search → correct response, no HITL, log entry appears
□ Turn 2: Note saving → HITL approval gate fires, approve → note saved
□ Turn 2b: Note saving → reject → note NOT saved, agent acknowledges
□ Turn 3: Calendar lookup → no HITL, fast response
□ Turn 4: Memory recall → ChromaDB returns Turn 2 fact without re-searching
□ Turn 5: Low-confidence query → feedback fires, thumbs down → SQLite row added
□ Logs: agent_calls.jsonl has entry for every LLM call
□ Log viewer: correct totals, per-node breakdown
□ Summarization: force > 2000 tokens, verify state["summary"] is populated
□ Gradio: approve/reject buttons visible only during HITL, hidden otherwise
□ Streaming: response appears token by token (if implemented)
```