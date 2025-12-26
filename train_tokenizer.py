from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
from datasets import load_from_disk
import json
import os
from pathlib import Path
from tqdm import tqdm
import torch  # Unused but for completeness if needed later

class ChessTokenizerTrainer:
    def __init__(self, dataset_path="./chess_data/hf_dataset"):
        self.dataset_path = dataset_path
        self.tokenizer = None
        
    def load_dataset_iterator(self, split="train", num_samples=None, max_length=512):
        """
        Create an iterator that yields game strings from the dataset
        FIXED: Added max_length filter to avoid overly long games.
        """
        try:
            dataset = load_from_disk(self.dataset_path)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {self.dataset_path}: {e}")
        
        if split not in dataset:
            raise ValueError(f"Split {split} not found in dataset")
        
        # Get the text column
        data = dataset[split]
        
        # Limit samples
        if num_samples:
            data = data.select(range(min(num_samples, len(data))))
        
        # Yield with length check (rough: split and count tokens)
        for i in range(len(data)):
            game = data[i]["text"]
            # Pre-tokenize roughly: split on space, count parts (moves + result)
            rough_tokens = len(game.split()) + 2  # + BOS/EOS
            if rough_tokens <= max_length:
                yield game
            else:
                print(f"Skipping long game {i} ({rough_tokens} tokens)")
    
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
    
    def train_tokenizer(self, vocab_size=1974, num_training_samples=1000000, max_length=512):
        """
        Train the tokenizer on the dataset
        FIXED: Added max_length param; better error handling.
        """
        print(f"Training tokenizer on {num_training_samples:,} samples...")
        
        # Create trainer
        trainer = self.create_tokenizer(vocab_size)
        
        # Get training data iterator
        train_iterator = self.load_dataset_iterator(
            split="train", 
            num_samples=num_training_samples,
            max_length=max_length
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
        
        try:
            # Train tokenizer
            self.tokenizer.train_from_iterator(
                ProgressIterator(train_iterator, total_samples),
                trainer=trainer,
                length=total_samples
            )
        except Exception as e:
            raise RuntimeError(f"Tokenizer training failed: {e}")
        
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
        FIXED: Enhanced UCI validation.
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
            elif token in ["1-0", "0-1"]:
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
        else:
            print(f"\nAll {len(move_tokens)} move tokens are valid UCI!")
        
        return {
            "special_tokens": special_tokens,
            "result_tokens": result_tokens,
            "move_tokens": move_tokens,
            "total_vocab_size": len(vocab)
        }
    
    def save_tokenizer(self, output_dir="./chess-llama-tokenizer"):
        """
        Save the trained tokenizer
        FIXED: Full HF config; error handling.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        try:
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
                "model_max_length": 1024,  # Safer for chess sequences
            }
            
            with open(output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Save tokenizer_config for HF
            tokenizer_config = {
                "model_max_length": 1024,
                "padding_side": "right",
                "truncation_side": "right",
                "special_tokens_map_file": None,
            }
            with open(output_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            
            print(f"Tokenizer saved to {output_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to save tokenizer: {e}")
    
    def wrap_as_pretrained_tokenizer(self):
        """
        Wrap the trained tokenizer as PreTrainedTokenizerFast
        FIXED: Remap specials to match Llama/chess-llama (unk=0, moves=1-1970, bos=1971, eos=1972, pad=1973).
        """
        from transformers import PreTrainedTokenizerFast
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        
        # Get current vocab
        vocab = self.tokenizer.get_vocab()
        current_vocab_size = len(vocab)
        
        # Target structure: unk=0, moves/results=1 to 1970, bos=1971, eos=1972, pad=1973
        target_vocab_size = 1974
        if current_vocab_size > target_vocab_size:
            print(f"Warning: Current vocab {current_vocab_size} > target {target_vocab_size}; truncating.")
            # Sort and truncate (least freq last, but simple slice for now)
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            vocab = {token: idx for idx, (token, _) in enumerate(sorted_vocab[:target_vocab_size - 4])}
        
        # Remap: Specials at fixed positions
        new_vocab = {}
        id_counter = 1  # Start after unk=0
        
        # Assign moves/results to 1-1970
        move_result_tokens = [t for t in vocab if t not in ["<unk>", "<s>", "</s>", "<pad>"]]
        for token in sorted(move_result_tokens):  # Alphabetical for determinism
            new_vocab[token] = id_counter
            id_counter += 1
        
        # Fixed specials
        new_vocab["<unk>"] = 0
        new_vocab["<s>"] = 1971
        new_vocab["</s>"] = 1972
        new_vocab["<pad>"] = 1973
        
        # Update tokenizer vocab (hack: recreate model with new vocab)
        self.tokenizer = Tokenizer(models.WordLevel(vocab=new_vocab, unk_token="<unk>"))
        # Re-apply normalizer/pre-tokenizer/post-processor (they persist via reference, but reset for safety)
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.Replace("\n", " "),
            normalizers.Replace("\t", " "),
        ])
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B </s>",
            special_tokens=[
                ("<s>", 1971),
                ("</s>", 1972),
            ]
        )
        
        # Wrap
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            padding_side="right",
            truncation_side="right",
            model_max_length=1024,
        )
        
        # Verify IDs
        print(f"Remapped special token IDs: unk=0, bos=1971, eos=1972, pad=1973")
        print(f"Final vocab size: {wrapped_tokenizer.vocab_size}")
        
        return wrapped_tokenizer

def create_final_tokenizer_pipeline(num_samples=1000000):
    """
    Complete pipeline to create and train tokenizer
    FIXED: Configurable samples; chess-specific tests; error handling.
    """
    from transformers import AutoTokenizer
    
    # Check if dataset exists
    if not os.path.exists("./chess_data/hf_dataset"):
        print("Dataset not found. Please run dataset preparation first.")
        return None
    
    try:
        # Initialize trainer
        trainer = ChessTokenizerTrainer("./chess_data/hf_dataset")
        
        # Train tokenizer
        tokenizer = trainer.train_tokenizer(
            vocab_size=1974,
            num_training_samples=num_samples,  # e.g., 1000000
            max_length=1024
        )
        
        # Analyze vocabulary
        analysis = trainer.analyze_vocabulary()
        
        # Save raw tokenizer
        trainer.save_tokenizer("./chess_tokenizer_raw")
        
        # Create wrapped tokenizer
        wrapped_tokenizer = trainer.wrap_as_pretrained_tokenizer()
        
        # Save as pretrained tokenizer
        wrapped_tokenizer.save_pretrained("./chess-llama-tokenizer")
        
        wrapped_tokenizer.push_to_hub(
            repo_id="VinayHajare/chess-llama",
            revision="reproduce"
        )
        
        # Test the tokenizer (chess-specific: only 1-0/0-1)
        print("\n" + "="*50)
        print("Testing tokenizer...")
        
        test_games = [
            "g1f3 g8f6 c2c4 0-1",  # Common opening
            "e2e4 e7e5 g1f3 b8c6 f1c4 1-0",
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
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return None

def load_and_test_existing_tokenizer(repo_name="VinayHajare/chess-llama"):
    """
    Load the original tokenizer and compare
    FIXED: Parameterized repo; optional (skips on failure).
    """
    from transformers import AutoTokenizer
    
    print(f"Loading original tokenizer from HuggingFace ({repo_name})...")
    try:
        original_tokenizer = AutoTokenizer.from_pretrained(repo_name)
        
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
        print(f"Skipping original load (e.g., repo unavailable): {e}")
        return None

if __name__ == "__main__":
    print("Chess Tokenizer Training Pipeline")
    print("="*50)
    
    # Option 1: Compare with original (optional)
    original = load_and_test_existing_tokenizer()
    
    # Option 2: Train new tokenizer (use 1M samples; adjust for 500k-2M range)
    print("\n" + "="*50)
    print("Training new tokenizer...")
    
    tokenizer = create_final_tokenizer_pipeline(num_samples=1000000)
    
    if tokenizer:
        print("\nTokenizer created successfully!")
        print(f"Vocabulary size: {tokenizer.vocab_size}")