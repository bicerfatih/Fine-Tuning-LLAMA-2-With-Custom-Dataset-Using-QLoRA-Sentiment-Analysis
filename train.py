# Import the necessary classes and functions
from transformers import AdamW  # Import a standard optimizer
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig


# Define the training function that accepts model, tokenizer, and datasets
def train(model, tokenizer, train_data, eval_data):
    # Configure LoRA parameters for efficient transformer tuning
    peft_config = LoraConfig(
        lora_alpha=16,  # Scale of the learned rank decomposition
        lora_dropout=0.1,  # Dropout rate in the LoRA layers
        r=64,  # Rank of the parameter matrices
        bias="none",  # No bias in LoRA layers
        task_type="CAUSAL_LM",  # Type of model being fine-tuned
    )

    # Set up training arguments that control the training process
    training_arguments = TrainingArguments(
        output_dir="logs",  # Directory for saving logs
        num_train_epochs=10,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size per device
        gradient_accumulation_steps=8,  # Number of steps to accumulate gradients before updating model weights
        optim="adamw_hf",  # Optimizer used during training
        save_steps=0,  # Model is not saved automatically during training
        logging_steps=20,  # Log metrics every 20 steps
        learning_rate=2e-4,  # Initial learning rate
        weight_decay=0.001,  # L2 penalty for regularization
        fp16=False,  # Disable mixed-precision training
        bf16=False,  # Disable bfloat16 precision training
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        max_steps=-1,  # Train indefinitely until epochs complete
        warmup_ratio=0.03,  # Fraction of total steps for learning rate warmup
        group_by_length=True,  # Group examples by length for efficient batching
        lr_scheduler_type="cosine",  # Use cosine learning rate scheduler
        report_to="tensorboard",  # Reporting metrics to TensorBoard
        evaluation_strategy="epoch"  # Evaluate at the end of each epoch
    )

    # Initialize the trainer with specified configurations and datasets
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,  # Disable input packing
        max_seq_length=1024,  # Maximum sequence length for model inputs
    )

    # Execute the training process
    trainer.train()

    # Save the trained model to specified directory
    output_dir = "results/trained-model"
    trainer.save_model(output_dir)
    pass  # End of the train function
