Healthy AI model directory.

Training creates:

healthy_ai_best.pth
healthy_ai_float32.onnx
class_names.json
metrics.json
dataset_split.json
training_history.csv
confusion_matrix.png

Quantization creates:

healthy_ai_int8.onnx

Run:

python train_model.py
python quantize_model.py