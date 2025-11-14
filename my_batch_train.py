from ultralytics import YOLO

results = []

if __name__ == '__main__':
    for i in range(8):
        model_path = f"my_model/yolo11-maf{i}.yaml"
        model = YOLO(model_path)
        
        metrics = model.train(
            data="my_dataset/Basketball_Detection.yaml",  # path to dataset YAML
            epochs=64,  # number of training epochs
            imgsz=640,  # training image size
            device=0  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
            # amp=False,  # 关闭自动混合精度
        )
        
        # 假设metrics有mAP50, mAP50-95等字段
        try:
            results.append({
                "model": model_path,
                "metrics": metrics.results_dict if hasattr(metrics, "results_dict") else metrics
            })
        except:
            print(f"Recording Metrics FAILED {i}")
    
    # 输出对比结果
    print("\nValidation Results Comparison:")
    for res in results:
        print(f"Model: {res['model']}")
        for k, v in res["metrics"].items():
            print(f"  {k}: {v}")
        print("-" * 30)
