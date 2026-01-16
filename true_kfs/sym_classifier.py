import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

class ShapeClassifier:
    def __init__(self, use_rf=True):
        if use_rf:
            self.clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            self.clf = KNeighborsClassifier(n_neighbors=3)
        self.label_to_name = {}
        
    def _extract_features(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None
            
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        cnt = max(contours, key=cv2.contourArea)
        
        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        
        features = np.concatenate([
            hu,
            [
                area / (w * h) if w * h > 0 else 0,
                perimeter / area if area > 0 else 0,
                w / h if h > 0 else 0,
                area / hull_area if hull_area > 0 else 0,
                len(approx),
                perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0,
            ]
        ])
        
        return features
    
    def _load_data(self, data_dir):
        X, y = [], []
        
        for img_path in sorted(Path(data_dir).glob('*.png')):
            feat = self._extract_features(img_path)
            if feat is not None:
                label = img_path.stem.split('_')[0]
                
                if label not in self.label_to_name:
                    self.label_to_name[label] = len(self.label_to_name)
                
                X.append(feat)
                y.append(self.label_to_name[label])
        
        return np.array(X), np.array(y)
    
    def train(self, train_dir):
        X_train, y_train = self._load_data(train_dir)
        
        if len(X_train) < 100:
            print(f"Small dataset detected ({len(X_train)} samples). Use dataset_generator.py to create more training data.")
            print("Training anyway, but accuracy may be limited...")
        
        self.clf.fit(X_train, y_train)
        print(f"Trained on {len(X_train)} samples, {len(self.label_to_name)} classes")
        
        return len(X_train), len(self.label_to_name)
    
    def evaluate(self, data_dir, split_name="test"):
        X, y_true = self._load_data(data_dir)
        
        if len(X) == 0:
            print(f"No data found in {data_dir}")
            return None
        
        y_pred = self.clf.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n{split_name.upper()} Results:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Samples: {len(X)}")
        
        name_to_label = {v: k for k, v in self.label_to_name.items()}
        target_names = [name_to_label[i] for i in sorted(name_to_label.keys())]
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        return accuracy
    
    def predict(self, img_path):
        feat = self._extract_features(img_path)
        if feat is None:
            return None, 0.0
        
        probs = self.clf.predict_proba([feat])[0]
        label_idx = np.argmax(probs)
        confidence = probs[label_idx]
        
        name_to_label = {v: k for k, v in self.label_to_name.items()}
        return name_to_label[label_idx], confidence
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.clf, self.label_to_name), f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.clf, self.label_to_name = pickle.load(f)
    
    def visualize_prediction(self, test_img_path, original_symbols_dir, save_path=None):
        predicted_name, confidence = self.predict(test_img_path)
        
        if predicted_name is None:
            print("Prediction failed")
            return
        
        test_img = cv2.imread(str(test_img_path))
        if test_img is None:
            print(f"Cannot load test image: {test_img_path}")
            return
        
        original_path = None
        for img_file in Path(original_symbols_dir).glob('*.png'):
            if img_file.stem.split('_')[-1] == predicted_name or img_file.stem == predicted_name:
                original_path = img_file
                break
        
        if original_path is None:
            print(f"Cannot find original symbol for: {predicted_name}")
            return
        
        original_img = cv2.imread(str(original_path))
        
        h1, w1 = test_img.shape[:2]
        h2, w2 = original_img.shape[:2]
        max_h = max(h1, h2)
        
        if h1 < max_h:
            pad = (max_h - h1) // 2
            test_img = cv2.copyMakeBorder(test_img, pad, max_h-h1-pad, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
        if h2 < max_h:
            pad = (max_h - h2) // 2
            original_img = cv2.copyMakeBorder(original_img, pad, max_h-h2-pad, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
        
        gap = np.ones((max_h, 50, 3), dtype=np.uint8) * 255
        combined = np.hstack([test_img, gap, original_img])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Test Image -> {predicted_name} ({confidence:.2%})"
        cv2.putText(combined, text, (10, 30), font, 0.7, (0, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, combined)
            print(f"Visualization saved to: {save_path}")
        else:
            cv2.imshow('Prediction Result', combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def predict_and_visualize(self, img_path, save_path=None):
        img_original = cv2.imread(str(img_path))
        if img_original is None:
            print(f"Could not load image: {img_path}")
            return None, 0.0, None
        
        feat = self._extract_features(img_path)
        if feat is None:
            print("Could not extract features")
            return None, 0.0, None
        
        probs = self.clf.predict_proba([feat])[0]
        name_to_label = {v: k for k, v in self.label_to_name.items()}
        
        top_3_indices = np.argsort(probs)[-3:][::-1]
        predictions = [(name_to_label[idx], probs[idx]) for idx in top_3_indices]
        
        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY) if len(img_original.shape) == 3 else img_original
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_with_contour = img_original.copy()
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(img_with_contour, [cnt], -1, (0, 255, 0), 3)
        
        h, w = img_original.shape[:2]
        viz_width = w + 400
        viz_height = max(h, 300)
        visualization = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255
        
        visualization[:h, :w] = img_with_contour
        
        text_x = w + 20
        cv2.putText(visualization, "Predictions:", (text_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        for i, (symbol, conf) in enumerate(predictions):
            y_pos = 80 + i * 80
            
            cv2.putText(visualization, f"{i+1}. {symbol}", (text_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.putText(visualization, f"{conf*100:.1f}%", (text_x, y_pos + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            bar_width = int(300 * conf)
            bar_color = (0, 200, 0) if i == 0 else (200, 200, 0) if i == 1 else (200, 100, 0)
            cv2.rectangle(visualization, (text_x, y_pos + 35), 
                         (text_x + bar_width, y_pos + 50), bar_color, -1)
            cv2.rectangle(visualization, (text_x, y_pos + 35), 
                         (text_x + 300, y_pos + 50), (200, 200, 200), 2)
        
        if save_path:
            cv2.imwrite(save_path, visualization)
            print(f"Visualization saved to: {save_path}")
        
        return predictions[0][0], predictions[0][1], visualization

if __name__ == "__main__":
    clf = ShapeClassifier()
    
    clf.train('dataset/train/')
    
    clf.evaluate('dataset/val/', split_name='validation')
    
    clf.evaluate('dataset/test/', split_name='test')
    
    clf.save('sym_model.pkl')
