from sym_classifier import ShapeClassifier

classifier = ShapeClassifier()
classifier.load('sym_model.pkl')

# Simple prediction
symbol_name, confidence = classifier.predict('dataset/sym1_red_004.png')
print(f"Predicted: {symbol_name} (confidence: {confidence:.2f})")

# Visualize the result (shows test image and matched original side by side)
classifier.visualize_prediction(
    test_img_path='dataset/test/sym1_red_079.png',
    original_symbols_dir='symbols/',
    save_path=f'results/result_visualization.png'
)



