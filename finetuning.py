"""
Fine-tuning Script for Mistral 7B-Instruct
ANTI-OVERFITTING Configuration for 20K Q&A Dataset
Optimized for stable training with proper regularization
"""

import os
import json
import torch
import wandb
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model settings
    MODEL_NAME = "unsloth/mistral-7b-v0.3"
    MAX_SEQ_LENGTH = 2048
    DTYPE = None  # None for auto detection
    LOAD_IN_4BIT = True
    
    # Alternative 4-bit pre-quantized models (faster download):
    # "unsloth/mistral-7b-bnb-4bit"
    # "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    # "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    
    # LoRA hyperparameters (REDUCED to prevent overfitting)
    LORA_R = 32  # Reduced from 64 - less capacity
    LORA_ALPHA = 64  # Reduced from 128
    LORA_DROPOUT = 0.2  # Increased from 0.1 - more dropout
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
    
    # Training hyperparameters (ANTI-OVERFITTING configuration)
    LEARNING_RATE = 2e-5  # Much more conservative
    WARMUP_RATIO = 0.1  # 10% warmup
    NUM_EPOCHS = 2  # Reduced from 3 - prevent overfitting
    PER_DEVICE_TRAIN_BATCH = 4
    PER_DEVICE_EVAL_BATCH = 8
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16
    MAX_GRAD_NORM = 0.5
    WEIGHT_DECAY = 0.1  # Increased from 0.01 - stronger regularization
    LR_SCHEDULER_TYPE = "cosine"
    
    # Early stopping settings (AGGRESSIVE to prevent overfitting)
    EARLY_STOPPING_PATIENCE = 2  # Reduced from 3 - stop faster
    EARLY_STOPPING_THRESHOLD = 0.001  # Increased threshold
    
    # Evaluation strategy
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 200
    SAVE_STEPS = 200
    LOGGING_STEPS = 25
    
    # Data split ratios (ANTI-OVERFITTING - larger validation set)
    TRAIN_RATIO = 0.80  # Reduced from 0.90 - 16,000 for training
    VAL_RATIO = 0.15    # Increased from 0.05 - 3,000 for validation
    TEST_RATIO = 0.05   # 1,000 for testing
    
    # Paths
    FR_DATA_PATH = "final_data_fr_premium.jsonl"
    EN_DATA_PATH = "final_data_eng_premium.jsonl"
    OUTPUT_DIR = "./mistral-7b-pm-expert"
    
    # W&B settings
    WANDB_PROJECT = "mistral-7b-pm-finetuning-v3"
    WANDB_RUN_NAME = "mistral-7b-qlora-bilingual-v3"

# ============================================================================
# DATA QUALITY DIAGNOSTICS
# ============================================================================

def diagnose_data_quality(fr_path, en_path):
    """Comprehensive data quality check"""
    print("\n" + "="*70)
    print("🔍 DATA QUALITY DIAGNOSTICS")
    print("="*70)
    
    def analyze_file(path, lang):
        data = []
        lengths = []
        user_questions = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    data.append(item)
                    
                    user_text = item['messages'][0]['content']
                    assistant_text = item['messages'][1]['content']
                    
                    total_length = len(user_text) + len(assistant_text)
                    lengths.append(total_length)
                    user_questions.append(user_text.lower().strip())
                    
                except Exception as e:
                    print(f"⚠️  Error on line {idx}: {e}")
        
        # Analysis
        print(f"\n📊 {lang} Dataset:")
        print(f"  Total examples: {len(data):,}")
        print(f"  Average length: {np.mean(lengths):.0f} chars")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        print(f"  Median length: {np.median(lengths):.0f} chars")
        
        # Check for duplicates
        question_counts = Counter(user_questions)
        duplicates = sum(1 for count in question_counts.values() if count > 1)
        print(f"  Duplicate questions: {duplicates}")
        
        # Length distribution
        too_short = sum(1 for l in lengths if l < 50)
        too_long = sum(1 for l in lengths if l > 3000)
        very_long = sum(1 for l in lengths if l > 5000)
        
        print(f"  Very short (<50 chars): {too_short}")
        print(f"  Long (>3000 chars): {too_long}")
        print(f"  Very long (>5000 chars): {very_long}")
        
        if very_long > 0:
            print(f"  ⚠️  Warning: {very_long} examples may exceed MAX_SEQ_LENGTH")
        
        return data, lengths
    
    fr_data, fr_lengths = analyze_file(fr_path, "French")
    en_data, en_lengths = analyze_file(en_path, "English")
    
    # Combined statistics
    all_lengths = fr_lengths + en_lengths
    print(f"\n📈 Combined Statistics:")
    print(f"  Total examples: {len(fr_data) + len(en_data):,}")
    print(f"  Average length: {np.mean(all_lengths):.0f} chars")
    print(f"  Std deviation: {np.std(all_lengths):.0f} chars")
    
    # Recommend adjustments
    max_length = max(all_lengths)
    if max_length > 4000:
        print(f"\n💡 Recommendation: Some examples are very long ({max_length} chars)")
        print(f"   Consider MAX_SEQ_LENGTH adjustment or filtering")
    
    return fr_data, en_data

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_jsonl_data(file_path):
    """Load JSONL data"""
    print(f"📂 Loading {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping malformed line {idx}: {e}")
    print(f"✓ Loaded {len(data)} examples")
    return data

def format_conversations(examples):
    """Format conversations for Mistral Instruct"""
    texts = []
    for messages in examples['messages']:
        text = ""
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'user':
                text += f"<s>[INST] {content} [/INST]"
            elif role == 'assistant':
                text += f" {content}</s>"
        texts.append(text)
    return {"text": texts}

def prepare_datasets(config):
    """Load, combine, and split datasets with quality checks"""
    print("\n" + "="*70)
    print("📚 LOADING AND PREPARING DATASETS")
    print("="*70)
    
    # Run diagnostics first
    fr_data, en_data = diagnose_data_quality(config.FR_DATA_PATH, config.EN_DATA_PATH)
    
    # Add language tags for tracking
    for item in fr_data:
        item['language'] = 'fr'
    for item in en_data:
        item['language'] = 'en'
    
    # Combine datasets
    all_data = fr_data + en_data
    total_samples = len(all_data)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  French examples:  {len(fr_data):,}")
    print(f"  English examples: {len(en_data):,}")
    print(f"  Total examples:   {total_samples:,}")
    
    # Calculate training estimates
    steps_per_epoch = int(total_samples * config.TRAIN_RATIO / (config.PER_DEVICE_TRAIN_BATCH * config.GRADIENT_ACCUMULATION_STEPS))
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    num_evals = total_steps // config.EVAL_STEPS
    
    print(f"\n⏱️  Training Estimation:")
    print(f"  Steps per epoch:  ~{steps_per_epoch:,}")
    print(f"  Total steps:      ~{total_steps:,}")
    print(f"  Evaluations:      ~{num_evals}")
    print(f"  Est. time:        ~{total_steps * 2 / 3600:.1f} hours (at 2s/step)")
    
    # Create splits
    print(f"\n🔀 Data Splitting:")
    print(f"  Train: {config.TRAIN_RATIO*100:.0f}% ({int(total_samples * config.TRAIN_RATIO):,} examples)")
    print(f"  Val:   {config.VAL_RATIO*100:.0f}% ({int(total_samples * config.VAL_RATIO):,} examples)")
    print(f"  Test:  {config.TEST_RATIO*100:.0f}% ({int(total_samples * config.TEST_RATIO):,} examples)")
    
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        all_data, 
        test_size=config.TEST_RATIO,
        random_state=42,
        shuffle=True
    )
    
    # Second split: separate train and validation
    val_size = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        random_state=42,
        shuffle=True
    )
    
    print(f"\n✓ Split complete:")
    print(f"  Train:      {len(train_data):,} examples ({len(train_data)/total_samples*100:.1f}%)")
    print(f"  Validation: {len(val_data):,} examples ({len(val_data)/total_samples*100:.1f}%)")
    print(f"  Test:       {len(test_data):,} examples ({len(test_data)/total_samples*100:.1f}%)")
    
    # Language distribution check
    train_fr = sum(1 for x in train_data if x.get('language') == 'fr')
    train_en = sum(1 for x in train_data if x.get('language') == 'en')
    val_fr = sum(1 for x in val_data if x.get('language') == 'fr')
    val_en = sum(1 for x in val_data if x.get('language') == 'en')
    
    print(f"\n🌍 Language Distribution:")
    print(f"  Training Set:")
    print(f"    French:  {train_fr:,} ({train_fr/len(train_data)*100:.1f}%)")
    print(f"    English: {train_en:,} ({train_en/len(train_data)*100:.1f}%)")
    print(f"  Validation Set:")
    print(f"    French:  {val_fr:,} ({val_fr/len(val_data)*100:.1f}%)")
    print(f"    English: {val_en:,} ({val_en/len(val_data)*100:.1f}%)")
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Format for instruction tuning
    print("\n🔄 Formatting datasets...")
    train_dataset = train_dataset.map(
        format_conversations,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        format_conversations,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    test_dataset = test_dataset.map(
        format_conversations,
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Show sample
    print("\n🔍 Sample formatted training example:")
    print("-" * 70)
    sample_text = train_dataset[0]['text']
    print(sample_text[:500] + ("..." if len(sample_text) > 500 else ""))
    print("-" * 70)
    
    # Save test set for later evaluation
    test_dataset.save_to_disk(f"{config.OUTPUT_DIR}/test_dataset")
    print(f"\n💾 Test dataset saved to: {config.OUTPUT_DIR}/test_dataset")
    
    return train_dataset, val_dataset, test_dataset

# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer(config):
    """Initialize model and tokenizer with Unsloth"""
    print("\n" + "="*70)
    print("🚀 LOADING MODEL")
    print("="*70)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=config.LOAD_IN_4BIT,
        # token = "hf_...", # use one if using gated models
    )
    
    print("✓ Base model loaded")
    
    # Apply LoRA with anti-overfitting settings
    print("\n🔧 Applying LoRA configuration (Anti-Overfitting)...")
    print(f"  Rank (r):        {config.LORA_R} (reduced for regularization)")
    print(f"  Alpha:           {config.LORA_ALPHA}")
    print(f"  Dropout:         {config.LORA_DROPOUT} (increased to 0.2)")
    print(f"  Target modules:  {len(config.TARGET_MODULES)} modules")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        target_modules=config.TARGET_MODULES,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ LoRA applied successfully")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Trainable %:           {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer

# ============================================================================
# TRAINING
# ============================================================================

def train_model(config):
    """Main training function with anti-overfitting measures"""
    print("\n" + "="*70)
    print("🎯 STARTING FINE-TUNING (ANTI-OVERFITTING MODE)")
    print("="*70)
    
    # Initialize W&B
    print("\n📊 Initializing Weights & Biases...")
    wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={
            "model": config.MODEL_NAME,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "lora_dropout": config.LORA_DROPOUT,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.PER_DEVICE_TRAIN_BATCH,
            "gradient_accumulation": config.GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": config.PER_DEVICE_TRAIN_BATCH * config.GRADIENT_ACCUMULATION_STEPS,
            "weight_decay": config.WEIGHT_DECAY,
            "train_ratio": config.TRAIN_RATIO,
            "val_ratio": config.VAL_RATIO,
        }
    )
    print("✓ W&B initialized")
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(config)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Training arguments
    print("\n⚙️  Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        
        # Training
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        
        # Optimizer
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        max_grad_norm=config.MAX_GRAD_NORM,
        optim="adamw_8bit",
        
        # Logging and evaluation
        logging_steps=config.LOGGING_STEPS,
        eval_strategy=config.EVAL_STRATEGY,
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # W&B
        report_to="wandb",
        
        # Performance
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        
        # Misc
        seed=42,
        dataloader_num_workers=4,
    )
    
    # Early stopping callback with aggressive settings
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
    )
    
    # Initialize trainer with Unsloth optimizations
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=training_args,
        callbacks=[early_stopping],
        packing=False,
    )
    
    # Training info
    total_steps = len(train_dataset) // (config.PER_DEVICE_TRAIN_BATCH * config.GRADIENT_ACCUMULATION_STEPS) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    print(f"\n📈 Training Configuration Summary:")
    print(f"  Total training steps:      {total_steps:,}")
    print(f"  Warmup steps:              {warmup_steps:,}")
    print(f"  Evaluation every:          {config.EVAL_STEPS} steps")
    print(f"  Early stopping patience:   {config.EARLY_STOPPING_PATIENCE} evaluations")
    print(f"  Effective batch size:      {config.PER_DEVICE_TRAIN_BATCH * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate:             {config.LEARNING_RATE}")
    print(f"  Weight decay:              {config.WEIGHT_DECAY} (strong regularization)")
    print(f"  LoRA dropout:              {config.LORA_DROPOUT}")
    
    print("\n🎯 Anti-Overfitting Measures Active:")
    print("  ✓ Reduced LoRA rank (32 vs 64)")
    print("  ✓ Increased dropout (0.2 vs 0.1)")
    print("  ✓ Strong weight decay (0.1 vs 0.01)")
    print("  ✓ Lower learning rate (2e-5 vs 5e-5)")
    print("  ✓ Larger validation set (15% vs 5%)")
    print("  ✓ Fewer epochs (2 vs 3)")
    print("  ✓ Aggressive early stopping")
    
    # Train
    print("\n" + "="*70)
    print("🏋️  TRAINING IN PROGRESS")
    print("="*70)
    print("Monitor training at: https://wandb.ai")
    print("\n⚠️  IMPORTANT: Watch for train/val loss gap!")
    print("  Healthy gap: < 0.10")
    print("  Warning gap: 0.10 - 0.20")
    print("  Stop if gap: > 0.20")
    print()
    
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("="*70)
    test_results = trainer.evaluate(test_dataset)
    print(f"\n✓ Test Loss: {test_results['eval_loss']:.4f}")
    
    # Log to W&B
    wandb.log({
        "test_loss": test_results['eval_loss'],
        "final_train_loss": trainer.state.log_history[-2]['loss'] if len(trainer.state.log_history) > 1 else 0,
    })
    
    # Calculate and display overfitting metrics
    try:
        final_train_loss = trainer.state.log_history[-2]['loss']
        final_val_loss = test_results['eval_loss']
        overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 0
        
        print(f"\n📈 Overfitting Analysis:")
        print(f"  Final train loss: {final_train_loss:.4f}")
        print(f"  Final test loss:  {final_val_loss:.4f}")
        print(f"  Loss ratio:       {overfitting_ratio:.2f}x")
        
        if overfitting_ratio < 1.2:
            print(f"  Status: ✅ EXCELLENT - Minimal overfitting")
        elif overfitting_ratio < 1.5:
            print(f"  Status: ✅ GOOD - Acceptable generalization")
        elif overfitting_ratio < 2.0:
            print(f"  Status: ⚠️  FAIR - Some overfitting detected")
        else:
            print(f"  Status: ❌ POOR - Significant overfitting")
    except:
        pass
    
    # Save models
    print("\n" + "="*70)
    print("💾 SAVING MODELS")
    print("="*70)
    
    # Save LoRA adapter
    print("\n1️⃣  Saving LoRA adapter...")
    model.save_pretrained(f"{config.OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/lora_adapter")
    print(f"   ✓ Saved to: {config.OUTPUT_DIR}/lora_adapter")
    
    # Save merged 16-bit model
    print("\n2️⃣  Saving merged 16-bit model (for inference)...")
    model.save_pretrained_merged(
        f"{config.OUTPUT_DIR}/model_16bit",
        tokenizer,
        save_method="merged_16bit"
    )
    print(f"   ✓ Saved to: {config.OUTPUT_DIR}/model_16bit")
    
    # Save GGUF quantized model
    print("\n3️⃣  Saving GGUF quantized model (for llama.cpp)...")
    model.save_pretrained_gguf(
        f"{config.OUTPUT_DIR}/model_gguf",
        tokenizer,
        quantization_method="q4_k_m"
    )
    print(f"   ✓ Saved to: {config.OUTPUT_DIR}/model_gguf")
    
    wandb.finish()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel outputs saved in: {config.OUTPUT_DIR}/")
    print(f"  - lora_adapter/  (LoRA weights)")
    print(f"  - model_16bit/   (Merged model for inference)")
    print(f"  - model_gguf/    (Quantized for llama.cpp)")
    print(f"  - test_dataset/  (Test set for evaluation)")
    print("\n🧪 Next step: Run 'python test_mistral.py' to test your model!")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = Config()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Train model
    train_model(config)