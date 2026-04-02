"""Gradio UI with Chat, Import, and Settings tabs.

Uses chat_service for chat logic — no duplication with chat.py.
"""

import json
import os
import tempfile

import gradio as gr

from app.chat_service import chat as chat_service_fn
from app.database import ChatMessage, save_settings as save_db_settings
from app.embedder import get_embedding_function
from app.importer import (
    run_import_pipeline, add_source_embeddings, remove_source_embeddings,
    ZipValidationError, _safe_collection_name,
)
from app.sources import load_sources, toggle_source, delete_source


def create_ui(
    collection,
    session_factory,
    twin_slug: str,
    twin_name: str,
    system_prompt: str,
    rewrite_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    chromadb_client,
    data_dir: str,
    embedding_model: str,
    embedding_base_url: str = "http://localhost:11434/v1",
    embedding_api_key: str = "ollama",
) -> gr.Blocks:
    """Create the Gradio UI with Chat, Import, and Settings tabs."""

    has_twin = collection is not None and collection.count() > 0

    llm_settings = {
        "base_url": llm_base_url,
        "model": llm_model,
        "api_key": llm_api_key,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "embedding_api_key": embedding_api_key,
    }

    def _get_collection():
        """Get the ChromaDB collection with current embedding settings."""
        ef = get_embedding_function(
            llm_settings["embedding_model"],
            base_url=llm_settings["embedding_base_url"],
            api_key=llm_settings["embedding_api_key"],
        )
        collection_name = _safe_collection_name(twin_slug)
        try:
            return chromadb_client.get_collection(collection_name, embedding_function=ef)
        except Exception:
            return collection

    def user_submit(message: str, history: list[list[str | None]]):
        if not message or not message.strip():
            return history, message, ""
        return history + [[message, None]], "", ""

    def bot_respond(history: list[list[str | None]], mode: str):
        if not history or history[-1][1] is not None:
            return history, ""

        message = history[-1][0]

        current_collection = _get_collection()
        if current_collection is None or current_collection.count() == 0:
            history[-1][1] = "⚠️ No twin loaded. Go to the **Import** tab and upload your Facebook data first."
            return history, ""

        result = chat_service_fn(
            content=message,
            collection=current_collection,
            session_factory=session_factory,
            twin_slug=twin_slug,
            twin_name=twin_name,
            system_prompt=system_prompt,
            rewrite_prompt=rewrite_prompt,
            llm_base_url=llm_settings["base_url"],
            llm_model=llm_settings["model"],
            llm_api_key=llm_settings["api_key"],
            mode=mode,
        )

        history[-1][1] = result.content

        if result.error:
            return history, ""

        meta = result.retrieval_metadata
        chunks = result.retrieved_chunks or []

        if mode == "rewrite":
            lines = [
                f"**Rewrite (copy)** — style from **{meta['chunks']}** chunks "
                f"(avg match **{meta['avg_similarity']}**). Twin repeats your line in your voice."
            ]
        else:
            lines = [f"📊 Matched **{meta['chunks']}** chunks (avg similarity: **{meta['avg_similarity']}**)"]

        for i, chunk in enumerate(chunks, 1):
            sim = round(1 - chunk["distance"], 3)
            doc = chunk["document"].replace("\n", " → ")
            lines.append(f"{i}. `[{sim}]` {doc}")

        return history, "\n".join(lines)

    def import_fn(file, source_name, target_name, progress=gr.Progress()):
        if file is None:
            return "Please upload a .zip file.", _render_sources()

        def on_progress(msg):
            progress(0, desc=msg)

        try:
            result = run_import_pipeline(
                zip_path=file.name if hasattr(file, "name") else file,
                chromadb_client=chromadb_client,
                data_dir=data_dir,
                embedding_model=llm_settings["embedding_model"],
                target_name=target_name if target_name else None,
                source_name=source_name if source_name else "",
                on_progress=on_progress,
                embedding_base_url=llm_settings["embedding_base_url"],
                embedding_api_key=llm_settings["embedding_api_key"],
            )
            msg = (
                f"Source added! **{result['twin_name']}**\n\n"
                f"- Source chunks: {result['source_chunks']}\n"
                f"- Total embedded (all sources): {result['chunks_embedded']}\n\n"
                f"**Restart the app** to chat with updated data."
            )
            return msg, _render_sources()
        except ZipValidationError as e:
            return f"Error: {e}", _render_sources()
        except ValueError as e:
            return f"Error: {e}", _render_sources()
        except Exception as e:
            return f"Failed to build twin. Error: {e}", _render_sources()

    def _render_sources() -> str:
        sources = load_sources(data_dir, twin_slug)
        if not sources:
            return "_No sources imported yet._"

        lines = ["| Status | Name | Messages | Chunks | DMs | Groups | ID |",
                 "|--------|------|----------|--------|-----|--------|-----|"]
        for s in sources:
            status = "✅" if s.enabled else "⏸️"
            lines.append(
                f"| {status} | {s.name} | {s.target_messages} | {s.train_chunks} | "
                f"{s.dm_chats} | {s.group_chats} | `{s.id}` |"
            )

        total_msgs = sum(s.target_messages for s in sources if s.enabled)
        total_chunks = sum(s.train_chunks for s in sources if s.enabled)
        lines.append(f"\n**Enabled totals:** {total_msgs} messages, {total_chunks} chunks")
        return "\n".join(lines)

    def toggle_source_fn(source_id: str, progress=gr.Progress()):
        sid = source_id.strip()
        sources = load_sources(data_dir, twin_slug)
        target = next((s for s in sources if s.id == sid), None)
        if not target:
            return f"Source `{source_id}` not found.", _render_sources()

        new_state = not target.enabled
        toggle_source(data_dir, twin_slug, sid, new_state)

        if new_state:
            progress(0, desc="Adding source embeddings...")
            count = add_source_embeddings(data_dir, twin_slug, sid, chromadb_client, llm_settings["embedding_model"],
                                          embedding_base_url=llm_settings["embedding_base_url"],
                                          embedding_api_key=llm_settings["embedding_api_key"])
            action = f"Enabled source `{sid}`. Added {count} chunks."
        else:
            progress(0, desc="Removing source embeddings...")
            count = remove_source_embeddings(twin_slug, sid, chromadb_client, llm_settings["embedding_model"],
                                             embedding_base_url=llm_settings["embedding_base_url"],
                                             embedding_api_key=llm_settings["embedding_api_key"])
            action = f"Disabled source `{sid}`. Removed {count} chunks."
        return (
            f"{action}\n\n**Restart the app** to use updated data."
        ), _render_sources()

    def delete_source_fn(source_id: str, progress=gr.Progress()):
        sid = source_id.strip()
        progress(0, desc="Removing source embeddings...")
        count = remove_source_embeddings(twin_slug, sid, chromadb_client, llm_settings["embedding_model"],
                                         embedding_base_url=llm_settings["embedding_base_url"],
                                         embedding_api_key=llm_settings["embedding_api_key"])

        if not delete_source(data_dir, twin_slug, sid):
            return f"Source `{source_id}` not found.", _render_sources()

        return (
            f"Deleted source `{sid}`. Removed {count} chunks.\n\n"
            f"**Restart the app** to use updated data."
        ), _render_sources()

    def export_fn():
        with session_factory() as session:
            messages = (
                session.query(ChatMessage)
                .filter_by(twin_slug=twin_slug)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            if not messages:
                return None

            export = []
            for msg in messages:
                export.append({
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "retrieval_metadata": msg.retrieval_metadata,
                })

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="chat_export_"
        )
        json.dump(export, tmp, indent=2, ensure_ascii=False)
        tmp.close()
        return tmp.name

    def save_settings(base_url, model, api_key, emb_model, emb_base_url, emb_api_key):
        llm_settings["base_url"] = base_url.strip()
        llm_settings["model"] = model.strip()
        llm_settings["api_key"] = api_key.strip()
        llm_settings["embedding_model"] = emb_model.strip()
        llm_settings["embedding_base_url"] = emb_base_url.strip()
        llm_settings["embedding_api_key"] = emb_api_key.strip()
        save_db_settings(session_factory, {
            "llm_base_url": base_url.strip(),
            "llm_model": model.strip(),
            "llm_api_key": api_key.strip(),
            "embedding_model": emb_model.strip(),
            "embedding_base_url": emb_base_url.strip(),
            "embedding_api_key": emb_api_key.strip(),
        })
        return (
            f"Settings saved. Using **{model}** at `{base_url}`\n\n"
            f"Embedding model: **{emb_model}** at `{emb_base_url}`"
        )

    def clear_chat():
        return [], "", ""

    # Build UI
    with gr.Blocks(title="Digital Twins", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Digital Twins")

        with gr.Tab("Chat"):
            mode = gr.Radio(
                choices=[
                    ("Answer — normal chat with your twin", "answer"),
                    ("Rewrite (copy) — twin repeats your sentence in your voice", "rewrite"),
                ],
                value="answer",
                label="Mode",
            )
            chatbot = gr.Chatbot(
                label=f"Chatting with {twin_slug.replace('_', ' ').title()}",
                height=450,
                type="tuples",
            )
            quality_indicator = gr.Markdown(value="", visible=True)

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type a message...",
                    show_label=False,
                    scale=9,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("Clear", size="sm")
                export_btn = gr.Button("Export", size="sm")
                export_file = gr.File(label="Download", visible=False)
                export_btn.click(fn=export_fn, outputs=export_file).then(
                    fn=lambda f: gr.update(visible=f is not None),
                    inputs=export_file,
                    outputs=export_file,
                )

            msg_input.submit(
                fn=user_submit,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input, quality_indicator],
            ).then(
                fn=bot_respond,
                inputs=[chatbot, mode],
                outputs=[chatbot, quality_indicator],
            )
            send_btn.click(
                fn=user_submit,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input, quality_indicator],
            ).then(
                fn=bot_respond,
                inputs=[chatbot, mode],
                outputs=[chatbot, quality_indicator],
            )
            clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input, quality_indicator])

        with gr.Tab("Import"):
            gr.Markdown("### Import Data Source")
            gr.Markdown("Upload a Facebook data export (.zip). Each import becomes a separate source you can enable/disable.")

            with gr.Row():
                upload = gr.File(label="Upload .zip file", file_types=[".zip"])
            with gr.Row():
                source_name_input = gr.Textbox(
                    label="Source name",
                    placeholder="e.g. 'Facebook main export' or 'E2EE messages'",
                )
                target_input = gr.Textbox(
                    label="Target name (optional)",
                    placeholder="Leave empty to auto-detect",
                )
            import_btn = gr.Button("Import Source", variant="primary")
            import_output = gr.Markdown(label="Status")

            gr.Markdown("---")
            gr.Markdown("### Imported Sources")
            sources_table = gr.Markdown(value=_render_sources())

            gr.Markdown("### Manage Sources")
            with gr.Row():
                source_id_input = gr.Textbox(
                    label="Source ID",
                    placeholder="Paste source ID from table above",
                    scale=3,
                )
                toggle_btn = gr.Button("Enable/Disable", scale=1)
                delete_btn = gr.Button("Delete", variant="stop", scale=1)

            manage_output = gr.Markdown()

            import_btn.click(
                fn=import_fn,
                inputs=[upload, source_name_input, target_input],
                outputs=[import_output, sources_table],
            )
            toggle_btn.click(
                fn=toggle_source_fn,
                inputs=[source_id_input],
                outputs=[manage_output, sources_table],
            )
            delete_btn.click(
                fn=delete_source_fn,
                inputs=[source_id_input],
                outputs=[manage_output, sources_table],
            )

        with gr.Tab("Settings"):
            gr.Markdown("### LLM Settings")
            gr.Markdown("Change the model without restarting. Takes effect on next message.")
            settings_base_url = gr.Textbox(
                label="LLM Base URL",
                value=llm_settings["base_url"],
                placeholder="http://localhost:11434/v1",
            )
            settings_model = gr.Textbox(
                label="Model",
                value=llm_settings["model"],
                placeholder="kimi-k2.5:cloud",
            )
            settings_api_key = gr.Textbox(
                label="API Key",
                value=llm_settings["api_key"],
                type="password",
            )

            gr.Markdown("### Embedding Settings")
            gr.Markdown("Uses OpenAI-compatible `/v1/embeddings` API. Works with OpenAI, Ollama, OpenRouter, etc.")
            settings_emb_model = gr.Textbox(
                label="Embedding Model",
                value=llm_settings["embedding_model"],
                placeholder="text-embedding-3-small",
            )
            settings_emb_base_url = gr.Textbox(
                label="Embedding Base URL",
                value=llm_settings["embedding_base_url"],
                placeholder="http://localhost:11434/v1",
            )
            settings_emb_api_key = gr.Textbox(
                label="Embedding API Key",
                value=llm_settings["embedding_api_key"],
                type="password",
            )

            save_btn = gr.Button("Save Settings", variant="primary")
            settings_output = gr.Markdown()
            save_btn.click(
                fn=save_settings,
                inputs=[settings_base_url, settings_model, settings_api_key,
                        settings_emb_model, settings_emb_base_url, settings_emb_api_key],
                outputs=settings_output,
            )

    return demo
