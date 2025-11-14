from ultralytics import YOLO
import os

models = [
    "best/train28.y/weights/best.pt",
    "best/train29.0/weights/best.pt",
    "best/train31.1/weights/best.pt",
    "best/train30.2/weights/best.pt",
    "best/train32.3/weights/best.pt",
    "best/train37.4/weights/best.pt",
    "best/train36.5/weights/best.pt",
    "best/train41.6/weights/best.pt",
    "best/train42.7/weights/best.pt",
    # 添加更多模型路径
]

results = []

if __name__ == '__main__':
    for model_path in models:
        print(f"Validating {model_path}...")
        model = YOLO(model_path)
        metrics = model.val(
            data="my_dataset/Basketball_Detection.yaml",
            imgsz=640,
            device=0
        )
        # 假设metrics有mAP50, mAP50-95等字段
        results.append({
            "model": model_path,
            "metrics": metrics.results_dict if hasattr(metrics, "results_dict") else metrics
        })

    # 输出对比结果
    print("\nValidation Results Comparison:")
    for res in results:
        print(f"Model: {res['model']}")
        for k, v in res["metrics"].items():
            print(f"  {k}: {v}")
        print("-" * 30)