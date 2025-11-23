"""
Fine-tuning Script for Mistral 7B-Instruct
Handles data loading, splitting, and training with W&B monitoring
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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model settings
    MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    
    # LoRA hyperparameters (optimized for 20K dataset)
    LORA_R = 64  # Higher capacity for better learning
    LORA_ALPHA = 128  # 2x rank
    LORA_DROPOUT = 0.1  # Slight regularization
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
    
    # Training hyperparameters (optimized for 20K dataset)
    LEARNING_RATE = 2e-4  # Higher LR safe for 20K samples
    WARMUP_RATIO = 0.05  # 5% warmup for stability
    NUM_EPOCHS = 3  # 3 epochs sufficient for 20K samples
    PER_DEVICE_TRAIN_BATCH = 4  # Increased batch size
    PER_DEVICE_EVAL_BATCH = 8
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER_TYPE = "cosine"
    
    # Early stopping settings (relaxed for 20K dataset)
    EARLY_STOPPING_PATIENCE = 3  # 3 evaluations is enough
    EARLY_STOPPING_THRESHOLD = 0.0005  # Smaller threshold
    
    # Evaluation strategy (adjusted for dataset size)
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 200  # More frequent evals with 20K samples
    SAVE_STEPS = 200
    LOGGING_STEPS = 25
    
    # Data split ratios (optimized for 20K samples)
    TRAIN_RATIO = 0.90  # 18,000 for training
    VAL_RATIO = 0.05    # 1,000 for validation
    TEST_RATIO = 0.05   # 1,000 for testing
    
    # Paths
    FR_DATA_PATH = "final_data_fr.jsonl"
    EN_DATA_PATH = "final_data_eng.jsonl"
    OUTPUT_DIR = "./mistral-7b-pm-expert"
    
    # W&B settings
    WANDB_PROJECT = "mistral-7b-pm-finetuning"
    WANDB_RUN_NAME = "mistral-7b-qlora-bilingual"

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
    """Load, combine, and split datasets"""
    print("\n" + "="*70)
    print("📚 LOADING AND PREPARING DATASETS")
    print("="*70)
    
    # Load both datasets
    fr_data = load_jsonl_data(config.FR_DATA_PATH)
    en_data = load_jsonl_data(config.EN_DATA_PATH)
    
    # Add language tags for tracking
    for item in fr_data:
        item['language'] = 'fr'
    for item in en_data:
        item['language'] = 'en'
    
    # Combine datasets
    all_data = fr_data + en_data
    total_samples = len(all_data)
    print(f"\n📊 Dataset Statistics:")
    print(f"  French examples:  {len(fr_data):,}")
    print(f"  English examples: {len(en_data):,}")
    print(f"  Total examples:   {total_samples:,}")
    
    # Calculate approximate training time
    steps_per_epoch = total_samples * config.TRAIN_RATIO // (config.PER_DEVICE_TRAIN_BATCH * config.GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    print(f"\n⏱️  Training Estimation:")
    print(f"  Steps per epoch:  ~{steps_per_epoch:,}")
    print(f"  Total steps:      ~{total_steps:,}")
    print(f"  Evaluations:      ~{total_steps // config.EVAL_STEPS}")
    
    # Create splits
    print(f"\n🔀 Splitting data:")
    print(f"  Train: {config.TRAIN_RATIO*100:.0f}%")
    print(f"  Val:   {config.VAL_RATIO*100:.0f}%")
    print(f"  Test:  {config.TEST_RATIO*100:.0f}%")
    
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
    print(f"\n🌍 Training Set Language Distribution:")
    print(f"  French:  {train_fr:,} ({train_fr/len(train_data)*100:.1f}%)")
    print(f"  English: {train_en:,} ({train_en/len(train_data)*100:.1f}%)")
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Format for instruction tuning
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
    print(train_dataset[0]['text'][:400] + "...")
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
        dtype=None,
        load_in_4bit=config.LOAD_IN_4BIT,
    )
    
    print("✓ Base model loaded")
    
    # Apply LoRA
    print("\n🔧 Applying LoRA configuration...")
    print(f"  Rank (r):        {config.LORA_R}")
    print(f"  Alpha:           {config.LORA_ALPHA}")
    print(f"  Dropout:         {config.LORA_DROPOUT}")
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
    """Main training function"""
    print("\n" + "="*70)
    print("🎯 STARTING FINE-TUNING")
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
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.PER_DEVICE_TRAIN_BATCH,
            "gradient_accumulation": config.GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": config.PER_DEVICE_TRAIN_BATCH * config.GRADIENT_ACCUMULATION_STEPS,
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
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        args=training_args,
        callbacks=[early_stopping],
        packing=False,
    )
    
    # Training info
    total_steps = len(train_dataset) // (config.PER_DEVICE_TRAIN_BATCH * config.GRADIENT_ACCUMULATION_STEPS) * config.NUM_EPOCHS
    print(f"\n📈 Training configuration:")
    print(f"  Total training steps:      {total_steps:,}")
    print(f"  Evaluation every:          {config.EVAL_STEPS} steps")
    print(f"  Early stopping patience:   {config.EARLY_STOPPING_PATIENCE} evaluations")
    print(f"  Effective batch size:      {config.PER_DEVICE_TRAIN_BATCH * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate:             {config.LEARNING_RATE}")
    print(f"  Warmup steps:              ~{int(total_steps * config.WARMUP_RATIO)}")
    
    # Train
    print("\n" + "="*70)
    print("🏋️  TRAINING IN PROGRESS")
    print("="*70)
    print("Monitor training at: https://wandb.ai")
    print()
    
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("="*70)
    test_results = trainer.evaluate(test_dataset)
    print(f"\n✓ Test Loss: {test_results['eval_loss']:.4f}")
    
    wandb.log({"test_loss": test_results['eval_loss']})
    
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

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = Config()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Train model
    train_model(config)