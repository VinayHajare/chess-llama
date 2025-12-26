from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
from datasets import load_from_disk
import json
import os
from pathlib import Path
from tqdm import tqdm

class ChessTokenizerTrainer:
    def __init__(self, dataset_path="./chess_data/hf_dataset"):
        self.dataset_path = dataset_path
        self.tokenizer = None
        
    def load_dataset_iterator(self, split="train", num_samples=None):
        """
        Create an iterator that yields game strings from the dataset
        """
        dataset = load_from_disk(self.dataset_path)
        
        if split not in dataset:
            raise ValueError(f"Split {split} not found in dataset")
        
        # Get the text column
        data = dataset[split]
        
        # Return iterator
        if num_samples:
            data = data.select(range(min(num_samples, len(data))))
        
        for i in range(len(data)):
            yield data[i]["text"]
    
    def create_tokenizer(self, vocab_size=1974):
        """
        Create and train a WordLevel tokenizer on the chess dataset
        """
        print("Initializing WordLevel tokenizer...")
        
        # Initialize a WordLevel tokenizer
        self.tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        
        # Add normalizer
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.Replace("\n", " "),
            normalizers.Replace("\t", " "),
        ])
        
        # Use Whitespace pre-tokenizer (UCI moves are space-separated)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Initialize trainer
        # Special tokens must be in the order we want them in vocabulary
        special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
        
        trainer = trainers.WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=True
        )
        
        return trainer
    
    def train_tokenizer(self, vocab_size=1974, num_training_samples=1000000):
        """
        Train the tokenizer on the dataset
        """
        print(f"Training tokenizer on {num_training_samples:,} samples...")
        
        # Create trainer
        trainer = self.create_tokenizer(vocab_size)
        
        # Get training data iterator
        train_iterator = self.load_dataset_iterator(
            split="train", 
            num_samples=num_training_samples
        )
        
        # Count total samples for progress bar
        total_samples = min(
            num_training_samples, 
            len(load_from_disk(self.dataset_path)["train"])
        )
        
        # Wrap iterator with tqdm for progress
        class ProgressIterator:
            def __init__(self, iterator, total):
                self.iterator = iterator
                self.total = total
                self.pbar = tqdm(total=total, desc="Training tokenizer")
                
            def __iter__(self):
                for item in self.iterator:
                    yield item
                    self.pbar.update(1)
                self.pbar.close()
        
        # Train tokenizer
        self.tokenizer.train_from_iterator(
            ProgressIterator(train_iterator, total_samples),
            trainer=trainer,
            length=total_samples
        )
        
        # Add post-processor for BOS/EOS
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B </s>",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>")),
            ]
        )
        
        print(f"Tokenizer training complete!")
        print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
        
        return self.tokenizer
    
    def analyze_vocabulary(self):
        """
        Analyze the trained vocabulary
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        vocab = self.tokenizer.get_vocab()
        
        # Count different types of tokens
        move_tokens = []
        result_tokens = []
        special_tokens = []
        
        for token, idx in vocab.items():
            if token in ["<unk>", "<s>", "</s>", "<pad>"]:
                special_tokens.append((token, idx))
            elif token in ["1-0", "0-1", "1/2-1/2"]:
                result_tokens.append((token, idx))
            else:
                move_tokens.append((token, idx))
        
        print(f"\nVocabulary Analysis:")
        print(f"Special tokens: {len(special_tokens)}")
        for token, idx in sorted(special_tokens, key=lambda x: x[1]):
            print(f"  {idx}: {token}")
        
        print(f"\nResult tokens: {len(result_tokens)}")
        for token, idx in sorted(result_tokens, key=lambda x: x[1]):
            print(f"  {idx}: {token}")
        
        print(f"\nMove tokens: {len(move_tokens)}")
        
        # Show first 20 move tokens
        print("First 20 move tokens:")
        for token, idx in sorted(move_tokens[:20], key=lambda x: x[1]):
            print(f"  {idx}: {token}")
        
        # Check if all moves are valid UCI
        invalid_moves = []
        for token, _ in move_tokens:
            if len(token) != 4:
                invalid_moves.append(token)
            elif not (token[0] in 'abcdefgh' and token[1] in '12345678' and
                     token[2] in 'abcdefgh' and token[3] in '12345678'):
                invalid_moves.append(token)
        
        if invalid_moves:
            print(f"\nWarning: {len(invalid_moves)} invalid UCI moves in vocabulary:")
            print(invalid_moves[:10])
        
        return {
            "special_tokens": special_tokens,
            "result_tokens": result_tokens,
            "move_tokens": move_tokens,
            "total_vocab_size": len(vocab)
        }
    
    def save_tokenizer(self, output_dir="./chess-llama-tokenizer"):
        """
        Save the trained tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save(str(output_dir / "tokenizer.json"))
        
        # Create config file
        config = {
            "tokenizer_type": "WordLevel",
            "vocab_size": self.tokenizer.get_vocab_size(),
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "model_max_length": 512,
        }
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Tokenizer saved to {output_dir}")
    
    def wrap_as_pretrained_tokenizer(self):
        """
        Wrap the trained tokenizer as PreTrainedTokenizerFast
        """
        from transformers import PreTrainedTokenizerFast
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        # Wrap the tokenizer
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            padding_side="right",
            truncation_side="right",
            model_max_length=512,
        )
        
        # Set token IDs to match original model structure
        vocab = self.tokenizer.get_vocab()
        
        # The original tokenizer has:
        # unk: 0, regular tokens: 1-1970, bos: 1971, eos: 1972, pad: 1973
        
        # We'll map our learned vocabulary to this structure
        # First, let's see what IDs we have
        current_special_ids = {
            "<unk>": vocab.get("<unk>", 0),
            "<s>": vocab.get("<s>", 0),
            "</s>": vocab.get("</s>", 0),
            "<pad>": vocab.get("<pad>", 0),
        }
        
        print(f"Current token IDs: {current_special_ids}")
        
        return wrapped_tokenizer

def create_final_tokenizer_pipeline():
    """
    Complete pipeline to create and train tokenizer
    """
    import torch
    from transformers import AutoTokenizer
    
    # Check if dataset exists
    if not os.path.exists("./chess_data/hf_dataset"):
        print("Dataset not found. Please run dataset preparation first.")
        return None
    
    # Initialize trainer
    trainer = ChessTokenizerTrainer("./chess_data/hf_dataset")
    
    # Train tokenizer
    tokenizer = trainer.train_tokenizer(
        vocab_size=1974,
        num_training_samples=1000000  # Use 1M games for training
    )
    
    # Analyze vocabulary
    analysis = trainer.analyze_vocabulary()
    
    # Save raw tokenizer
    trainer.save_tokenizer("./chess_tokenizer_raw")
    
    # Create wrapped tokenizer
    wrapped_tokenizer = trainer.wrap_as_pretrained_tokenizer()
    
    # Save as pretrained tokenizer
    wrapped_tokenizer.save_pretrained("./chess-llama-tokenizer")
    
    # Test the tokenizer
    print("\n" + "="*50)
    print("Testing tokenizer...")
    
    test_games = [
        "1-0 g1f3 g8f6 c2c4",
        "0-1 e2e4 e7e5 g1f3",
        "1/2-1/2 d2d4 d7d5 c2c4",
    ]
    
    for game in test_games:
        print(f"\nGame: {game}")
        encoded = wrapped_tokenizer(game)
        print(f"Tokens: {encoded['input_ids']}")
        print(f"Decoded: {wrapped_tokenizer.decode(encoded['input_ids'])}")
    
    # Verify special token IDs
    print("\nSpecial token IDs:")
    print(f"unk_token_id: {wrapped_tokenizer.unk_token_id}")
    print(f"bos_token_id: {wrapped_tokenizer.bos_token_id}")
    print(f"eos_token_id: {wrapped_tokenizer.eos_token_id}")
    print(f"pad_token_id: {wrapped_tokenizer.pad_token_id}")
    
    return wrapped_tokenizer

def load_and_test_existing_tokenizer():
    """
    Load the original tokenizer and compare
    """
    from transformers import AutoTokenizer
    
    print("Loading original tokenizer from HuggingFace...")
    try:
        original_tokenizer = AutoTokenizer.from_pretrained("VinayHajare/chess-llama")
        
        print(f"\nOriginal tokenizer info:")
        print(f"Vocab size: {original_tokenizer.vocab_size}")
        print(f"Special tokens: {original_tokenizer.special_tokens_map}")
        
        # Test encoding
        test_game = "1-0 g1f3 g8f6 c2c4"
        encoded = original_tokenizer(test_game)
        print(f"\nOriginal tokenizer on '{test_game}':")
        print(f"Input IDs: {encoded['input_ids']}")
        print(f"Tokens: {original_tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")
        
        return original_tokenizer
        
    except Exception as e:
        print(f"Error loading original tokenizer: {e}")
        return None

if __name__ == "__main__":
    print("Chess Tokenizer Training Pipeline")
    print("="*50)
    
    # Option 1: Compare with original
    original = load_and_test_existing_tokenizer()
    
    # Option 2: Train new tokenizer
    print("\n" + "="*50)
    print("Training new tokenizer...")
    
    tokenizer = create_final_tokenizer_pipeline()
    
    if tokenizer:
        print("\nTokenizer created successfully!")
        print(f"Vocabulary size: {tokenizer.vocab_size}")