import torch
from transformers import (
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
import os
import gc

def setup_colab():
    """Setup Colab environment"""    
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

def load_datasets(data_path="./chess_data/hf_dataset"):
    """Load preprocessed datasets"""
    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
    else:
        # Load from HuggingFace Hub as fallback
        dataset = load_dataset("VinayHajare/chess-llama-dataset")
    
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    
    return dataset

def prepare_model_and_tokenizer():
    """Load or create model and tokenizer"""
    from transformers import AutoTokenizer
    
    # Try to load existing tokenizer, else create new
    tokenizer_path = "./chess-llama-tokenizer"
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("VinayHajare/chess-llama", revision="reproduce-branch")
    
    # Create model
    from model_config import create_chess_llama_model
    model, config = create_chess_llama_model()
    
    return model, tokenizer, config

def train_model_colab():
    """Main training function optimized for Colab"""
    setup_colab()
    
    # Load datasets
    print("Loading datasets...")
    dataset = load_datasets()
    
    # Prepare model and tokenizer
    print("Preparing model and tokenizer...")
    model, tokenizer, config = prepare_model_and_tokenizer()
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments optimized for Colab T4/GPU
    training_args = TrainingArguments(
        output_dir="./chess-llama-output",
        hub_model_id="VinayHajare/chess-llama",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,    # Reduced for Colab memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,    # Effective batch size = 4 * 4 = 16
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,                        # Mixed precision for Colab
        gradient_checkpointing=True,      # Save memory
        optim="adamw_torch",
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        save_total_limit=2,
        push_to_hub=True,
        hub_revision="reproduce-branch"                
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    save_path = "./chess-llama-final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    trainer.push_to_hub("VinayHajare/chess-llama", revision="reproduce-branch")
        
    print("Training complete!")
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return trainer

if __name__ == "__main__":
    # Start training
    trainer = train_model_colab()