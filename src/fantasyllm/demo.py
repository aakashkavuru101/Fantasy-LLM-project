"""Gradio demo interface for Fantasy LLM."""

import gradio as gr
from .generate import main as generate_text
from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


def interface(prompt: str) -> str:
    """Interface function for Gradio."""
    try:
        # Capture the generated text
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Redirect stdout to capture the generated text
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            generate_text(prompt)
        
        generated_text = output_buffer.getvalue().strip()
        logger.info("Demo generation completed", prompt_length=len(prompt))
        return generated_text
        
    except Exception as e:
        logger.error("Demo generation failed", error=str(e))
        return f"Error generating story: {str(e)}"


def main() -> None:
    """Launch the Gradio demo."""
    logger.info("Starting Gradio demo", 
                host=settings.gradio_server_name,
                port=settings.gradio_server_port)
    
    with gr.Blocks(
        title="FantasyLLM ✨",
        theme=gr.themes.Soft(),
    ) as demo:
        
        gr.Markdown("# FantasyLLM ✨")
        gr.Markdown("Production-ready LoRA fine-tuned Llama-2 for fantasy short stories")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    lines=5,
                    placeholder="Tell me a fantasy story about a brave knight...",
                    label="Story Prompt",
                    info="Enter your fantasy story prompt here"
                )
                
                generate_btn = gr.Button("Generate Story ✨", variant="primary")
                
                gr.Markdown("### Settings")
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=settings.max_new_tokens,
                    step=10,
                    label="Max New Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=settings.temperature,
                    step=0.1,
                    label="Temperature"
                )
            
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    lines=15,
                    label="Generated Fantasy Story",
                    interactive=False
                )
        
        generate_btn.click(
            fn=interface,
            inputs=prompt_input,
            outputs=output_text,
        )
        
        # Add example prompts
        gr.Examples(
            examples=[
                ["Tell me a fantasy story about a dragon who lost their fire"],
                ["Write a tale of a wizard's apprentice who accidentally turned their master into a frog"],
                ["Create a story about an enchanted forest where the trees can speak"],
                ["Tell the story of a brave knight on a quest to find the legendary crystal of wisdom"],
            ],
            inputs=prompt_input,
        )
    
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
        show_error=settings.environment != "production",
    )

if __name__ == "__main__":
    main()