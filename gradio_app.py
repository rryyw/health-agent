import os
import base64
import mimetypes

import gradio as gr
from dotenv import load_dotenv

from app.agent.health_agent import agent

load_dotenv()

TITLE = "大学生健康管理智能体"
DESCRIPTION = "输入健康问题，可选上传图片。"


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def build_user_content(message: str, image_path: str | None):
    message = (message or "").strip()

    if image_path:
        text = message or "请识别这张图片，并从大学生健康管理角度给出分析和建议。"
        return [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image_path)
                }
            }
        ]

    return message


def normalize_text(content) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    return str(content)


def has_image(content) -> bool:
    if not isinstance(content, list):
        return False

    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            return True
    return False


def get_role(message) -> str | None:
    if isinstance(message, dict):
        role = message.get("role") or message.get("type")
    else:
        role = getattr(message, "type", None) or getattr(message, "role", None)

    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return role


def get_content(message):
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def format_user_display(content) -> str:
    text = normalize_text(content)
    if has_image(content):
        return f"{text}\n\n[已上传图片]" if text else "[已上传图片]"
    return text


def to_chatbot_history(messages) -> list[dict]:
    history = []

    for message in messages:
        role = get_role(message)
        content = get_content(message)

        if role == "user":
            history.append(
                {
                    "role": "user",
                    "content": format_user_display(content)
                }
            )
        elif role == "assistant":
            text = normalize_text(content)
            if text.strip():
                history.append(
                    {
                        "role": "assistant",
                        "content": text
                    }
                )

    return history


def chat(message, image_path, ui_history, messages_state):
    ui_history = ui_history or []
    messages_state = messages_state or []

    message = (message or "").strip()
    if not message and not image_path:
        return ui_history, messages_state, "", None

    user_message = {
        "role": "user",
        "content": build_user_content(message, image_path)
    }

    try:
        request_messages = messages_state + [user_message]
        result = agent.invoke({"messages": request_messages})

        if isinstance(result, dict) and "messages" in result:
            messages_state = result["messages"]
        else:
            messages_state = request_messages

        ui_history = to_chatbot_history(messages_state)

    except Exception as e:
        ui_history = ui_history + [
            {
                "role": "user",
                "content": format_user_display(user_message["content"])
            },
            {
                "role": "assistant",
                "content": f"请求失败：{e}"
            }
        ]

    return ui_history, messages_state, "", None


def clear_chat():
    return [], [], "", None


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    chatbot = gr.Chatbot(label="对话", height=520)
    ui_history_state = gr.State(value=[])
    messages_state = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=4):
            message_box = gr.Textbox(
                label="问题",
                placeholder="例如：我22岁，男，178cm，80kg，想减脂，请帮我制定一周计划",
                lines=3
            )
        with gr.Column(scale=2):
            image_box = gr.Image(
                type="filepath",
                label="上传图片（可选）"
            )

    with gr.Row():
        send_button = gr.Button("发送", variant="primary")
        clear_button = gr.Button("清空会话")

    send_button.click(
        chat,
        inputs=[message_box, image_box, ui_history_state, messages_state],
        outputs=[chatbot, messages_state, message_box, image_box]
    ).then(
        lambda history: history,
        inputs=[chatbot],
        outputs=[ui_history_state]
    )

    message_box.submit(
        chat,
        inputs=[message_box, image_box, ui_history_state, messages_state],
        outputs=[chatbot, messages_state, message_box, image_box]
    ).then(
        lambda history: history,
        inputs=[chatbot],
        outputs=[ui_history_state]
    )

    clear_button.click(
        clear_chat,
        inputs=[],
        outputs=[chatbot, ui_history_state, message_box, image_box]
    ).then(
        lambda: [],
        inputs=[],
        outputs=[messages_state]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False
    )