import requests
import os
import subprocess
import glob
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import chess.pgn
import io
import zipfile

class ChessDatasetProcessor:
    def __init__(self, output_dir="./chess_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_lichess_database(self, start_year=2020, start_month=6, end_year=2025, end_month=11):
        """
        Download Lichess elite database for specified years and months from https://database.nikonoel.fr/
        Files are zip archives containing .pgn files.
        Download from June 2020 to December 2025
        """
        print("Downloading Lichess elite database...")
        base_url = "https://database.nikonoel.fr"
        raw_dir = os.path.join(self.output_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        for year in range(start_year, end_year + 1):
            month_start = start_month if year == start_year else 1
            month_end = end_month if year == end_year else 12
            for month in range(month_start, month_end + 1):
                filename = f"lichess_elite_{year}-{month:02d}.zip"
                url = f"{base_url}/{filename}"
                output_path = os.path.join(raw_dir, filename)
                # Resume check
                if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                    print(f"{filename} already exists and is non-empty; skipping.")
                    continue
                print(f"Downloading {filename}...")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    with open(output_path, 'wb') as f, tqdm(
                        desc=filename,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for data in response.iter_content(chunk_size=1024):
                            size = f.write(data)
                            bar.update(size)
                except requests.exceptions.HTTPError:
                    print(f"Skipping unavailable {filename} (e.g., 404)")
                    if os.path.exists(output_path):
                        os.remove(output_path)  # Clean up partial
                    continue
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    continue
        print("Download complete!")

    def install_dependencies(self):
        """Install required system packages"""
        print("Installing dependencies...")
        try:
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "pgn-extract"], check=True)
        except subprocess.CalledProcessError:
            print("apt-get failed (likely no sudo/root). In Colab, run manually: !apt-get update && !apt-get install -y pgn-extract")
        try:
            subprocess.run(["pip", "install", "chess", "tqdm", "requests", "datasets"], check=True)
        except subprocess.CalledProcessError:
            print("pip install failed. Ensure running in an environment with pip access.")

    def extract_and_filter_games(self):
        """
        Extract games from PGN files (inside zip) and filter for checkmate endings.
        FIXED: Now checks Result header (1-0 or 0-1) instead of Termination.
        FIXED: Properly batches games without losing data.
        FIXED: Verifies checkmate via board simulation.
        FIXED: Robust PGN file finding.
        FIXED: Simplified PGN parsing loop (read_game directly, no _offset seek).
        FIXED: Result moved to end of formatted game (moves first, then result).
        Requires 30min for 60K checkmate games
        USE THIS FOR: Maximum control, custom filtering, smaller datasets
        """
        raw_dir = os.path.join(self.output_dir, "raw")
        processed_dir = os.path.join(self.output_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        batch_games = []
        batch_num = 0
        total_games = 0
        for filepath in sorted(glob.glob(os.path.join(raw_dir, "*.zip"))):
            print(f"Processing {os.path.basename(filepath)}...")
            # Extract zip file
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            # Robust PGN finding
            pgn_files = glob.glob(os.path.join(raw_dir, "*.pgn"))
            if not pgn_files:
                print(f"No PGN found after extracting {os.path.basename(filepath)}")
                continue
            pgn_path = pgn_files[0]
            # Parse PGN and filter for decisive games (checkmate endings)
            try:
                with open(pgn_path, 'r') as f:
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        # FIXED: Check Result from game.headers
                        # Checkmate games have result 1-0 (white wins) or 0-1 (black wins)
                        # Draws (1/2-1/2) are excluded
                        result = game.headers.get("Result", "")
                       
                        if result in ["1-0", "0-1"]:
                            # Simulate board to verify checkmate
                            board = game.board()
                            for move in game.mainline_moves():
                                board.push(move)
                            # FIXED: Verify checkmate
                            if board.is_checkmate():
                                uci_moves = [move.uci() for move in game.mainline_moves()]
                                # FIXED: Result at end
                                formatted_game = " ".join(uci_moves) + f" {result}"
                                batch_games.append(formatted_game)
                                total_games += 1
                                # FIXED: Save batch when it reaches 100k games
                                if len(batch_games) >= 100000:
                                    batch_num += 1
                                    batch_file = os.path.join(processed_dir, f"games_batch_{batch_num}.txt")
                                    with open(batch_file, 'w') as batch_f:
                                        batch_f.write('\n'.join(batch_games))
                                    print(f"Saved batch {batch_num} with {len(batch_games)} games (Total: {total_games})")
                                    batch_games = [] # Clear batch after saving
            except Exception as e:
                print(f"Error parsing {pgn_path}: {e}")
            # Clean up extracted PGN file
            if os.path.exists(pgn_path):
                os.remove(pgn_path)
        # FIXED: Save remaining games in final batch
        if batch_games:
            batch_num += 1
            batch_file = os.path.join(processed_dir, f"games_batch_{batch_num}.txt")
            with open(batch_file, 'w') as batch_f:
                batch_f.write('\n'.join(batch_games))
            print(f"Saved final batch {batch_num} with {len(batch_games)} games")
        print(f"Processing complete! Total checkmate games saved: {total_games}")
        return processed_dir

    def convert_to_uci_with_pgnextract(self):
        """
        OPTIMIZED: Fast C-based conversion using pgn-extract with proper batching.
        This is significantly faster than Python-based parsing (10-50x speedup).
       
        USE THIS FOR: Large datasets (millions of games), production pipelines
        Requires 1min for 60k checkmate games
        FIXED:
        - Single pgn-extract call for checkmate + UCI
        - Added batching support
        - Filter only checkmate games
        - Proper memory management
        - Better error handling
        - Robust PGN finding
        - Result moved to end of formatted game (moves first, then result).
        """
        pgn_extract_path = "/usr/games/pgn-extract"
        processed_dir = os.path.join(self.output_dir, "processed_uci")
        os.makedirs(processed_dir, exist_ok=True)
        raw_dir = os.path.join(self.output_dir, "raw")
       
        batch_games = []
        batch_num = 0
        total_games = 0
        for filepath in sorted(glob.glob(os.path.join(raw_dir, "*.zip"))):
            print(f"Converting {os.path.basename(filepath)} with pgn-extract...")
            # Extract zip file
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            # Robust PGN finding
            pgn_files = glob.glob(os.path.join(raw_dir, "*.pgn"))
            if not pgn_files:
                print(f"No PGN found after extracting {os.path.basename(filepath)}")
                continue
            pgn_path = pgn_files[0]
            # Single pgn-extract call: Filter checkmates + UCI output
            cmd = [
                pgn_extract_path,
                "--checkmate",      # Only true mates (decisive by definition)
                "-Wuci",            # UCI format (single line per game)
                "--notags",         # No headers
                "--novars",         # No variations
                "--nomovenumbers",  # No move nums
                pgn_path
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                uci_output = result.stdout
                # Parse: Each line is "e2e4 e7e5 ... 1-0"
                for line in uci_output.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5 and parts[-1] in ['1-0', '0-1']:  # Min ~4 moves + result
                            result_str = parts[-1]
                            moves = parts[:-1]
                            # FIXED: Result at end
                            formatted_game = " ".join(moves) + f" {result_str}"
                            batch_games.append(formatted_game)
                            total_games += 1
                            # Save batch when it reaches 100k games
                            if len(batch_games) >= 100000:
                                batch_num += 1
                                batch_file = os.path.join(processed_dir, f"games_batch_{batch_num}.txt")
                                with open(batch_file, 'w') as batch_f:
                                    batch_f.write('\n'.join(batch_games))
                                print(f"Saved batch {batch_num} with {len(batch_games)} games (Total: {total_games})")
                                batch_games = []
            except subprocess.CalledProcessError as e:
                print(f"pgn-extract failed for {os.path.basename(filepath)}: {e}")
            except FileNotFoundError:
                print(f"pgn-extract not found at {pgn_extract_path}. Install via !apt-get install pgn-extract")
            # Clean up
            if os.path.exists(pgn_path):
                os.remove(pgn_path)
        # Save remaining games in final batch
        if batch_games:
            batch_num += 1
            batch_file = os.path.join(processed_dir, f"games_batch_{batch_num}.txt")
            with open(batch_file, 'w') as batch_f:
                batch_f.write('\n'.join(batch_games))
            print(f"Saved final batch {batch_num} with {len(batch_games)} games")
        print(f"Processing complete! Total checkmate games saved: {total_games} (~{total_games * 0.3 / 1000000:.1f}M expected from elite DB)")
        return processed_dir

    def create_huggingface_dataset(self, processed_dir, ext="txt", batch_size=5, val_split=0.05):
        """
        Create train/validation split and prepare for HuggingFace with memory-efficient streaming.
        Processes files in batches to stay within RAM limits.
        
        Args:
            processed_dir: Directory containing the text files
            ext: File extension (default "txt")
            batch_size: Number of files to process at once (tune based on RAM)
            val_split: Validation split ratio (default 0.05 for 5%)
        """
        from datasets import Dataset, DatasetDict
        import random
        
        pattern = os.path.join(processed_dir, f"*.{ext}")
        filepaths = sorted(glob.glob(pattern))
        
        if not filepaths:
            raise ValueError(f"No .{ext} files found in {processed_dir}")
        
        print(f"Found {len(filepaths)} files")
        
        # Shuffle files for better train/val distribution
        random.shuffle(filepaths)
        
        # Count total games first (lightweight pass)
        print("Counting total games...")
        total_games = 0
        for filepath in filepaths:
            with open(filepath, 'r') as f:
                total_games += sum(1 for line in f if line.strip())
        
        print(f"Total games: {total_games:,}")
        val_size = int(total_games * val_split)
        train_size = total_games - val_size
        
        print(f"Train size: {train_size:,}, Val size: {val_size:,}")
        
        # Generator functions for streaming
        def train_generator():
            games_yielded = 0
            for filepath in filepaths:
                if games_yielded >= train_size:
                    break
                with open(filepath, 'r') as f:
                    for line in f:
                        if games_yielded >= train_size:
                            break
                        line = line.strip()
                        if line:
                            yield {"text": line}
                            games_yielded += 1
        
        def val_generator():
            games_skipped = 0
            games_yielded = 0
            for filepath in filepaths:
                if games_yielded >= val_size:
                    break
                with open(filepath, 'r') as f:
                    for line in f:
                        if games_yielded >= val_size:
                            break
                        line = line.strip()
                        if line:
                            if games_skipped < train_size:
                                games_skipped += 1
                                continue
                            yield {"text": line}
                            games_yielded += 1
        
        # Create datasets from generators
        print("Creating train dataset...")
        train_dataset = Dataset.from_generator(train_generator)
        
        print("Creating validation dataset...")
        val_dataset = Dataset.from_generator(val_generator)
        
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        # Save to disk
        save_path = os.path.join(self.output_dir, "hf_dataset")
        print(f"Saving to disk: {save_path}")
        dataset.save_to_disk(save_path)
        
        # Push to hub
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub("VinayHajare/chess-llama-dataset")
        
        return dataset


    def run_full_pipeline(self, use_pgnextract=True):
        """
        Run complete dataset processing pipeline
       
        Args:
            use_pgnextract: If True, uses fast C-based pgn-extract (recommended for large datasets)
                          If False, uses Python-based parsing (more control, slower)
        """
        print("Starting Chess Llama dataset pipeline...")
        self.install_dependencies()
        self.download_lichess_database()
       
        if use_pgnextract:
            print("Using pgn-extract (fast C-based method)...")
            processed_dir = self.convert_to_uci_with_pgnextract()
            ext = "txt"  # Outputs .txt
        else:
            print("Using Python-based parsing (slower but more control)...")
            processed_dir = self.extract_and_filter_games()
            ext = "txt"
       
        dataset = self.create_huggingface_dataset(processed_dir, ext=ext)
        print("Pipeline completed!")
        print(f"Train games: {len(dataset['train'])}")
        print(f"Validation games: {len(dataset['validation'])}")
        return dataset

if __name__ == "__main__":
    processor = ChessDatasetProcessor()
   
    # For production with millions of games - use pgn-extract (FAST)
    processor.run_full_pipeline(use_pgnextract=True)
   
    # For smaller datasets or custom filtering - use Python method
    # processor.run_full_pipeline(use_pgnextract=False)