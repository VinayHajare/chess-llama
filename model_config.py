from transformers import LlamaConfig, LlamaForCausalLM
import torch

def create_chess_llama_model():
    config = LlamaConfig(
        vocab_size=1974,           # Valid moves + special tokens
        hidden_size=512,           # Model dimensions
        intermediate_size=1024,    # FFN dimensions
        num_hidden_layers=8,       # Number of layers
        num_attention_heads=8,     # Attention heads
        num_key_value_heads=8,     # Key/Value heads
        max_position_embeddings=512, # Max sequence length
        pad_token_id=1973,         
        bos_token_id=1971,         
        eos_token_id=1972,         
        hidden_act="silu",         # SiLU activation
        rms_norm_eps=1e-06,        
        attention_bias=False,
        use_cache=True,
        torch_dtype=torch.float32,
    )
    
    # Create model
    model = LlamaForCausalLM(config)
    
    # Verify parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected: 23,001,600")
    print(f"Difference: {total_params - 23001600}")
    
    return model, config

if __name__ == "__main__":
    model, config = create_chess_llama_model()
    print("\nModel Configuration:")
    print(config)
    
    # Save configuration
    config.save_pretrained("chess-llama-config")
    print("\nConfiguration saved to 'chess-llama-config'")