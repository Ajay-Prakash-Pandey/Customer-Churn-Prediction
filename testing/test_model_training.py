from src.pipeline.train_pipeline import TrainPipeline


def test_model_training():

    result = TrainPipeline().run_pipeline()

    print("\n========== MODEL TRAINING RESULT ==========")
    print(f"Best Model    : {result['best_model_name']}")
    print(f"Test Accuracy : {result['test_accuracy']:.4f}")
    print(f"Model Path    : {result['model_path']}")
    print("===========================================")


if __name__ == "__main__":
    test_model_training()
