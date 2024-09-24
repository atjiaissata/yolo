from ultralytics import YOLO, settings
import os
import mlflow
import tempfile


class YoloV10():

    def __init__(self, config):
        self.config = config
    
    def experiment(self):
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.config.project
    
    
    def mlflow_tracking(self):
        os.environ["MLFLOW_TRACKING_URI"] = self.config.tracking_uri

    def __enter__(self):
        self.start()
        return self
    
    def start(self):
        self.mlflow_tracking()
        self.experiment()
        mlflow.start_run()
        self.run_id = mlflow.active_run().info.run_id
        params = self.config.__dict__.copy()
        del params["tracking_uri"]
        del params["project"]
        mlflow.log_params(params)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        mlflow.end_run()   
    
    def fit(self):
        settings.update({"mlflow": True})
        
        with self:
            model = YOLO('yolov10n.pt')
            results = model.train(
                data=self.config.dataset,
                epochs=self.config.epoch,
                project=tempfile.gettempdir(),
                name=self.config.project,
                batch=self.config.batch_size,
                optimizer=self.config.optimizer,
                seed=self.config.seed,
                device="cpu",
                patience = self.config.patience,
                lr0=self.config.learning_rate,
                lrf=1,
                weight_decay=self.config.weight_decay,
                val=True,
                close_mosaic = 0, 
                amp=True,
                box=15
                        )