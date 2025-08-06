import gradio as gr
from .generate import main as generate

def interface(prompt):
    return generate(prompt)

def main():
    gr.Interface(
        fn=interface,
        inputs=gr.Textbox(lines=3, placeholder="Tell me a fantasy story…"),
        outputs=gr.Textbox(lines=10),
        title="FantasyLLM ✨",
        description="LoRA fine-tuned Llama-2 for fantasy short stories",
    ).launch(share=True)

if __name__ == "__main__":
    main()