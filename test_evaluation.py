"""
Test your fine-tuned Mistral 7B PM Expert GGUF model
Works with llama-cpp-python
"""

import subprocess
import os
import sys

# Configuration
GGUF_MODEL_PATH = "./mistral-7b-pm-expert/model_gguf/model-q4_k_m.gguf"
LLAMA_CPP_DIR = "./llama.cpp"

def check_llama_cpp():
    """Check if llama.cpp is available"""
    llama_cli = os.path.join(LLAMA_CPP_DIR, "llama-cli")
    
    if not os.path.exists(llama_cli):
        print("❌ llama-cli not found!")
        print(f"\nPlease make sure llama.cpp is compiled:")
        print(f"  cd {LLAMA_CPP_DIR}")
        print(f"  make")
        return False
    
    return llama_cli

def test_with_llama_cpp(question, max_tokens=512, temp=0.7):
    """Test model using llama.cpp CLI"""
    llama_cli = check_llama_cpp()
    
    if not llama_cli:
        return None
    
    # Format prompt in Mistral Instruct format
    prompt = f"<s>[INST] {question} [/INST]"
    
    # Build command
    cmd = [
        llama_cli,
        "-m", GGUF_MODEL_PATH,
        "-p", prompt,
        "-n", str(max_tokens),
        "-c", "2048",
        "--temp", str(temp),
        "--top-p", "0.9",
        "--repeat-penalty", "1.1",
        "-ngl", "0",  # Use CPU (0 GPU layers)
    ]
    
    print(f"🤖 Generating response...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Extract response
        output = result.stdout
        
        # Find the answer after [/INST]
        if "[/INST]" in output:
            answer = output.split("[/INST]")[-1].strip()
            # Clean up llama.cpp metadata
            answer = answer.split("\n\n")[0] if "\n\n" in answer else answer
            return answer
        
        return output
        
    except subprocess.TimeoutExpired:
        return "⚠️ Generation timed out (>2 minutes)"
    except Exception as e:
        return f"❌ Error: {e}"

def test_with_python_binding():
    """Test using llama-cpp-python"""
    try:
        from llama_cpp import Llama
        
        print("🚀 Loading model with llama-cpp-python...")
        
        llm = Llama(
            model_path=GGUF_MODEL_PATH,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0,  # CPU inference
            verbose=False
        )
        
        print("✓ Model loaded\n")
        
        def generate(question, max_tokens=512):
            prompt = f"<s>[INST] {question} [/INST]"
            
            print(f"🤖 Generating response...")
            
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["</s>", "[INST]"],
            )
            
            return output['choices'][0]['text'].strip()
        
        return generate
        
    except ImportError:
        print("⚠️ llama-cpp-python not installed")
        print("Install it with: pip install llama-cpp-python")
        return None

def run_test_suite(generate_func):
    """Run comprehensive PM test questions"""
    
    test_questions = [
        # French PM Questions
        {
            "lang": "🇫🇷 French",
            "questions": [
                "Qu'est-ce que la gestion de projet?",
                "Comment créer un plan de gestion de projet efficace?",
                "Quelles sont les phases du cycle de vie d'un projet?",
                "Comment gérer les risques dans un projet?",
                "Expliquez la méthode Agile et ses avantages.",
            ]
        },
        # English PM Questions
        {
            "lang": "🇬🇧 English",
            "questions": [
                "What is project management?",
                "What are the key components of a project charter?",
                "How do you manage stakeholder expectations?",
                "What is the critical path in project management?",
                "What is the difference between Agile and Waterfall?",
            ]
        }
    ]
    
    print("="*70)
    print("🧪 MISTRAL 7B PM EXPERT - TEST SUITE")
    print("="*70)
    print(f"\nModel: {GGUF_MODEL_PATH}")
    print(f"Format: GGUF Q4_K_M (4-bit quantized)")
    print(f"Size: 4.07 GB\n")
    
    for category in test_questions:
        print("\n" + "="*70)
        print(f"{category['lang']} PROJECT MANAGEMENT QUESTIONS")
        print("="*70)
        
        for i, question in enumerate(category['questions'], 1):
            print(f"\n{'─'*70}")
            print(f"Question {i}/{len(category['questions'])}:")
            print(f"{'─'*70}")
            print(f"Q: {question}\n")
            
            # Generate answer
            answer = generate_func(question)
            
            print(f"A: {answer}\n")
            print("─"*70)
            
            # Wait for user
            if i < len(category['questions']):
                user_input = input("\n➡️  Press Enter for next question (or 'q' to quit): ")
                if user_input.lower() == 'q':
                    return
    
    print("\n" + "="*70)
    print("✅ TEST SUITE COMPLETE!")
    print("="*70)

def interactive_mode(generate_func):
    """Interactive Q&A session"""
    print("="*70)
    print("💬 INTERACTIVE MODE")
    print("="*70)
    print("\n💡 Ask your PM questions (type 'quit' to exit)")
    print("Commands:")
    print("  - 'quit' or 'q': Exit")
    print("  - 'test': Run full test suite")
    print()
    
    while True:
        question = input("🤔 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if question.lower() == 'test':
            run_test_suite(generate_func)
            continue
        
        if not question:
            continue
        
        print()
        answer = generate_func(question)
        print(f"✓ Answer:\n{answer}\n")
        print("─"*70)

def main():
    """Main function"""
    print("="*70)
    print("🧪 MISTRAL 7B PM EXPERT - MODEL TESTING")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"\n❌ Model not found: {GGUF_MODEL_PATH}")
        print("\nPlease run the GGUF export script first.")
        return
    
    print(f"\n✓ Model found: {GGUF_MODEL_PATH}")
    
    # Get model size
    size_gb = os.path.getsize(GGUF_MODEL_PATH) / (1024**3)
    print(f"✓ Model size: {size_gb:.2f} GB\n")
    
    # Choose testing method
    print("Choose testing method:")
    print("  1. llama-cpp-python (recommended, Python API)")
    print("  2. llama.cpp CLI (direct binary)")
    print("  3. Quick single test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Python binding
        generate_func = test_with_python_binding()
        
        if generate_func is None:
            print("\n💡 Falling back to llama.cpp CLI...")
            choice = "2"
        else:
            print("\nChoose mode:")
            print("  1. Auto test suite")
            print("  2. Interactive Q&A")
            
            mode = input("\nEnter choice (1-2): ").strip()
            
            if mode == "1":
                run_test_suite(generate_func)
            else:
                interactive_mode(generate_func)
            return
    
    if choice == "2":
        # CLI method
        llama_cli = check_llama_cpp()
        
        if not llama_cli:
            print("\n❌ Cannot proceed without llama.cpp")
            print("\nSetup instructions:")
            print("  git clone https://github.com/ggerganov/llama.cpp")
            print("  cd llama.cpp")
            print("  make")
            return
        
        # Test question
        print("\n" + "="*70)
        print("🧪 QUICK TEST")
        print("="*70)
        
        test_q = "What are the key phases of project management?"
        print(f"\nQuestion: {test_q}\n")
        
        answer = test_with_llama_cpp(test_q)
        print(f"Answer:\n{answer}\n")
        
    elif choice == "3":
        # Quick test
        print("\n💡 Using Python binding for quick test...")
        generate_func = test_with_python_binding()
        
        if generate_func:
            question = "Qu'est-ce que la gestion de projet?"
            print(f"\nTest question: {question}\n")
            answer = generate_func(question)
            print(f"Answer:\n{answer}\n")
        else:
            print("❌ Cannot run quick test")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()