"""
Working GGUF export - saves merged model first, then converts from that location
"""

import os
import shutil
from unsloth import FastLanguageModel

# Configuration
OUTPUT_DIR = "./mistral-7b-pm-expert"
BASE_MODEL = "unsloth/mistral-7b-v0.3"
MAX_SEQ_LENGTH = 2048

def merge_and_save_lora():
    """Merge LoRA with base model and save in full precision"""
    import torch
    
    print("="*70)
    print("STEP 1: MERGING LORA WEIGHTS")
    print("="*70)
    
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint-1536")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"\n✓ Found checkpoint: {checkpoint_path}\n")
    
    # Load base model in full precision (not 4-bit)
    print("🚀 Loading base model in full precision...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Will use default (float32)
        load_in_4bit=False,  # IMPORTANT: Load in full precision, not 4-bit
    )
    
    # Load LoRA
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_path)
    print("✓ LoRA loaded\n")
    
    # Merge
    print("🔀 Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    print("✓ Merged\n")
    
    # Save merged model in full precision
    merged_dir = os.path.join(OUTPUT_DIR, "merged_model_f16")
    
    if os.path.exists(merged_dir):
        print(f"🗑️  Removing old merged model...")
        shutil.rmtree(merged_dir)
    
    # Convert to float16 to save space
    print(f"💾 Converting to float16 and saving merged model...")
    model = model.to(torch.float16)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print("✅ Merged model saved in float16!\n")
    
    return merged_dir

def convert_to_gguf_directly():
    """Convert using llama.cpp directly with python3 fix"""
    print("="*70)
    print("STEP 2: CONVERTING TO GGUF (using llama.cpp)")
    print("="*70)
    
    import subprocess
    import sys
    
    merged_dir = os.path.join(OUTPUT_DIR, "merged_model_f16")
    gguf_dir = os.path.join(OUTPUT_DIR, "model_gguf")
    
    if not os.path.exists(merged_dir):
        print(f"❌ Merged model not found: {merged_dir}")
        return False
    
    print(f"\n📂 Converting from: {merged_dir}\n")
    
    # Check if llama.cpp exists
    if not os.path.exists("llama.cpp"):
        print("📥 Cloning llama.cpp repository...")
        ret = subprocess.run("git clone https://github.com/ggerganov/llama.cpp", shell=True)
        if ret.returncode != 0:
            print("❌ Failed to clone llama.cpp")
            return False
        
        print("🔨 Building llama.cpp...")
        ret = subprocess.run("cd llama.cpp && make -j$(nproc)", shell=True)
        if ret.returncode != 0:
            print("❌ Failed to build llama.cpp")
            return False
    
    os.makedirs(gguf_dir, exist_ok=True)
    
    # Step 1: Convert to F16 GGUF
    print("💾 Step 1: Converting to GGUF (float16 format)...")
    print("   ⏱️  This will take ~3-5 minutes...\n")
    
    gguf_f16 = os.path.join(gguf_dir, "model-f16.gguf")
    
    # Use python3 explicitly in the command
    cmd1 = f"{sys.executable} llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile {gguf_f16} --outtype f16 --split-max-size 50G"
    
    print(f"Running: {cmd1}\n")
    ret = subprocess.run(cmd1, shell=True, capture_output=False, text=True)
    
    if ret.returncode != 0 or not os.path.exists(gguf_f16):
        print("❌ F16 conversion failed")
        return False
    
    size_gb = os.path.getsize(gguf_f16) / (1024**3)
    print(f"✅ F16 GGUF created: {size_gb:.2f} GB\n")
    
    # Step 2: Quantize to q4_k_m
    print("💾 Step 2: Quantizing to q4_k_m (4-bit)...")
    print("   ⏱️  This will take ~5-10 minutes...\n")
    
    gguf_q4 = os.path.join(gguf_dir, "model-q4_k_m.gguf")
    
    cmd2 = f"./llama.cpp/llama-quantize {gguf_f16} {gguf_q4} q4_k_m"
    
    print(f"Running: {cmd2}\n")
    ret = subprocess.run(cmd2, shell=True, capture_output=False, text=True)
    
    if ret.returncode != 0 or not os.path.exists(gguf_q4):
        print("❌ Quantization failed")
        return False
    
    size_gb = os.path.getsize(gguf_q4) / (1024**3)
    print(f"\n✅ Q4_K_M GGUF created: {size_gb:.2f} GB\n")
    
    return True

def alternative_gguf_method():
    """Alternative: Use llama.cpp directly"""
    print("\n" + "="*70)
    print("ALTERNATIVE METHOD: Using llama.cpp convert script directly")
    print("="*70)
    
    merged_dir = os.path.join(OUTPUT_DIR, "merged_model_16bit")
    gguf_dir = os.path.join(OUTPUT_DIR, "model_gguf")
    
    os.makedirs(gguf_dir, exist_ok=True)
    
    print(f"""
To convert manually using llama.cpp:

1. Install llama.cpp (if not already):
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make

2. Convert HF model to GGUF:
   python llama.cpp/convert_hf_to_gguf.py \\
     {merged_dir} \\
     --outfile {gguf_dir}/model-f16.gguf \\
     --outtype f16

3. Quantize to q4_k_m:
   ./llama.cpp/llama-quantize \\
     {gguf_dir}/model-f16.gguf \\
     {gguf_dir}/model-q4_k_m.gguf \\
     q4_k_m

This will create the final q4_k_m GGUF model (~4GB).
    """)

def main():
    print("="*70)
    print("🚀 MISTRAL 7B PM EXPERT - GGUF EXPORT")
    print("="*70)
    print()
    
    # Step 1: Merge LoRA
    merged_dir = merge_and_save_lora()
    
    if merged_dir is None:
        print("\n❌ Failed to merge LoRA")
        return
    
    print("\n✅ Step 1 Complete: LoRA merged successfully")
    print(f"   Merged model: {merged_dir}")
    
    # Step 2: Convert to GGUF
    print("\n" + "="*70)
    success = convert_to_gguf_directly()
    
    if not success:
        print("\n⚠️  Unsloth GGUF conversion failed")
        alternative_gguf_method()
        print("\n💡 You can use the merged_model_16bit for inference instead")
        print("   or follow the manual conversion steps above.")
    else:
        print("\n" + "="*70)
        print("✅ SUCCESS! GGUF EXPORT COMPLETE")
        print("="*70)
        
        gguf_dir = os.path.join(OUTPUT_DIR, "model_gguf")
        
        # List GGUF files
        if os.path.exists(gguf_dir):
            gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith('.gguf')]
            if gguf_files:
                print(f"\n📁 GGUF files created:")
                for gguf_file in gguf_files:
                    full_path = os.path.join(gguf_dir, gguf_file)
                    size_gb = os.path.getsize(full_path) / (1024**3)
                    print(f"   ✓ {gguf_file} ({size_gb:.2f} GB)")
        
        print(f"\n🚀 Usage with llama.cpp:")
        print(f"""
    ./llama-cli \\
      -m {gguf_dir}/model-q4_k_m.gguf \\
      -p "<s>[INST] Qu'est-ce que la gestion de projet? [/INST]" \\
      -n 512 \\
      --temp 0.7
        """)
        
        print("\n💡 Disk space:")
        print(f"   - merged_model_16bit: ~15 GB (can be deleted)")
        print(f"   - model_gguf:         ~4 GB (keep this)")
    
    print("\n✨ Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()