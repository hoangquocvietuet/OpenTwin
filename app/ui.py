"""Gradio UI with Chat and Import tabs.

Uses chat_service for chat logic — no duplication with chat.py.
"""

import json
import os
import tempfile

import gradio as gr

from app.chat_service import chat as chat_service_fn
from app.database import ChatMessage
from app.importer import run_import_pipeline, ZipValidationError


def create_ui(
    collection,
    session_factory,
    twin_slug: str,
    system_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    chromadb_client,
    data_dir: str,
    embedding_model: str,
) -> gr.Blocks:
    """Create the Gradio UI with Chat and Import tabs."""

    def chat_fn(message: str, history: list[dict]) -> str:
        """Handle chat messages via shared chat_service."""
        result = chat_service_fn(
            content=message,
            collection=collection,
            session_factory=session_factory,
            twin_slug=twin_slug,
            system_prompt=system_prompt,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )

        if result.error:
            return result.content

        quality = f"\n\n_Matched {result.retrieval_metadata['chunks']} chunks (avg similarity: {result.retrieval_metadata['avg_similarity']})_"
        return result.content + quality

    def import_fn(file, target_name, progress=gr.Progress()):
        """Handle file import."""
        if file is None:
            return "Please upload a .zip file."

        def on_progress(msg):
            progress(0, desc=msg)

        try:
            result = run_import_pipeline(
                zip_path=file.name if hasattr(file, "name") else file,
                chromadb_client=chromadb_client,
                data_dir=data_dir,
                embedding_model=embedding_model,
                target_name=target_name if target_name else None,
                on_progress=on_progress,
            )
            overwrite_note = " (previous twin data was overwritten)" if result.get("overwritten") else ""
            return (
                f"Twin ready! **{result['twin_name']}**{overwrite_note}\n\n"
                f"- Messages: {result['total_messages']}\n"
                f"- Chunks embedded: {result['chunks_embedded']}\n\n"
                f"Switch to the Chat tab to start talking!"
            )
        except ZipValidationError as e:
            return f"Error: {e}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Failed to build twin. Error: {e}"

    def export_fn():
        """Export chat history as JSON."""
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

    # Build UI
    with gr.Blocks(title="Digital Twins", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Digital Twins")

        with gr.Tab("Chat"):
            chatbot = gr.ChatInterface(
                fn=chat_fn,
                type="messages",
                title=None,
                description="Chat with your digital twin",
            )
            export_btn = gr.Button("Export Conversation", size="sm")
            export_file = gr.File(label="Download", visible=False)
            export_btn.click(fn=export_fn, outputs=export_file).then(
                fn=lambda f: gr.update(visible=f is not None),
                inputs=export_file,
                outputs=export_file,
            )

        with gr.Tab("Import"):
            gr.Markdown("### Import Facebook Messenger Data")
            gr.Markdown("Upload your Facebook data export (.zip) to build a twin.")
            upload = gr.File(label="Upload .zip file", file_types=[".zip"])
            target_input = gr.Textbox(
                label="Target name (optional)",
                placeholder="Leave empty to auto-detect from data",
            )
            import_btn = gr.Button("Build Twin", variant="primary")
            import_output = gr.Markdown(label="Status")
            import_btn.click(
                fn=import_fn,
                inputs=[upload, target_input],
                outputs=import_output,
            )

    return demo
