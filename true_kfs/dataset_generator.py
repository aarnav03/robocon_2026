import cv2
import numpy as np
from pathlib import Path
import random
import shutil

class DatasetGenerator:
    def __init__(self, symbol_dir, output_dir):
        self.symbol_dir = Path(symbol_dir)
        self.output_dir = Path(output_dir)
        
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
    def generate(self, samples_per_symbol=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
        
        symbol_files = sorted(self.symbol_dir.glob('*.png'))
        
        if not symbol_files:
            print(f"No symbol files found in {self.symbol_dir}")
            return
        
        n_train = int(samples_per_symbol * train_ratio)
        n_val = int(samples_per_symbol * val_ratio)
        n_test = samples_per_symbol - n_train - n_val
        
        print(f"Split: {n_train} train, {n_val} val, {n_test} test per symbol")
        print(f"Processing {len(symbol_files)} symbols...\n")
        
        for sym_file in symbol_files:
            sym_name = sym_file.stem.split('_')[-1]
            symbol = cv2.imread(str(sym_file), cv2.IMREAD_GRAYSCALE)
            
            if symbol is None:
                print(f"Warning: Could not load {sym_file}")
                continue
            
            samples = []
            for i in range(samples_per_symbol):
                angle = random.randint(0, 359)
                scale = random.uniform(0.7, 1.3)
                bg_color = random.choice(['red', 'blue'])
                
                img = self._create_sample(symbol, angle, scale, bg_color)
                samples.append((img, bg_color, i))
            
            random.shuffle(samples)
            
            train_samples = samples[:n_train]
            val_samples = samples[n_train:n_train + n_val]
            test_samples = samples[n_train + n_val:]
            
            self._save_samples(train_samples, sym_name, self.train_dir)
            self._save_samples(val_samples, sym_name, self.val_dir)
            self._save_samples(test_samples, sym_name, self.test_dir)
            
            print(f"Generated samples for '{sym_name}': {len(train_samples)} train, "
                  f"{len(val_samples)} val, {len(test_samples)} test")
        
        print(f"\n✓ Dataset created in {self.output_dir}/")
        print(f"  - Train: {self.train_dir}")
        print(f"  - Val: {self.val_dir}")
        print(f"  - Test: {self.test_dir}")
    
    def _save_samples(self, samples, sym_name, output_dir):
        for img, bg_color, idx in samples:
            filename = f"{sym_name}_{bg_color}_{idx:03d}.png"
            cv2.imwrite(str(output_dir / filename), img)
    
    def _create_sample(self, symbol, angle, scale, bg_color):
        h, w = symbol.shape
        
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(symbol, M, (w, h), borderValue=255)
        
        if bg_color == 'red':
            bg = np.full((h, w, 3), [0, 0, 255], dtype=np.uint8)
        else:
            bg = np.full((h, w, 3), [255, 0, 0], dtype=np.uint8)
        
        mask = rotated < 128
        bg[mask] = [0, 0, 0]
        
        return bg
    
    def get_dataset_stats(self):
        stats = {}
        for split_name, split_dir in [('train', self.train_dir), ('val', self.val_dir), ('test', self.test_dir)]:
            files = list(split_dir.glob('*.png'))
            stats[split_name] = len(files)
            
            # Count samples per class
            class_counts = {}
            for f in files:
                sym_name = f.stem.split('_')[0]
                class_counts[sym_name] = class_counts.get(sym_name, 0) + 1
            
            print(f"\n{split_name.upper()}: {len(files)} total samples")
            print(f"  Classes: {len(class_counts)}")
            if class_counts:
                print(f"  Samples per class: {list(class_counts.values())[:5]}...")
        
        return stats

if __name__ == "__main__":
    gen = DatasetGenerator('symbols/', 'dataset/')
    gen.generate(samples_per_symbol=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    print("\n" + "="*50)
    gen.get_dataset_stats()
