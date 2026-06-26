# MLOps Workflow Diagram

This workflow document shows the end-to-end process for training, tracking, and serving the Decision Tree model.

```text
      +----------------+       +--------------------+       +----------------------+  
      |                |       |                    |       |                      |  
      |  Data Loading   | ----> |  Preprocessing     | ----> |  Training Pipeline   |  
      |  (load_iris)   |       |  (StandardScaler)  |       |  (DecisionTree + ML) |  
      |                |       |                    |       |                      |  
      +-------+--------+       +---------+----------+       +----------+-----------+  
              |                          |                             |              
              v                          v                             v              
   +----------------------+   +----------------------+    +------------------------+  
   | train_test_split()   |   |  fit() on training   |    |  evaluate() on test   |  
   |  (train / test split)|   |  data                |    |  data                  |  
   +----------+-----------+   +----------+-----------+    +-----------+------------+  
              |                          |                             |              
              |                          |                             |              
              v                          v                             v              
   +----------------------+   +----------------------+    +------------------------+  
   | MLflow Logging       |   | Model Artifact stored|    |  Metrics stored in     |  
   |  params, metrics,    |   |  in mlruns/         |    |  MLflow run            |  
   |  and model           |   |                     |    |                        |  
   +----------+-----------+   +----------+-----------+    +-----------+------------+  
              |                          |                             |              
              v                          v                             v              
 +----------------------+   +----------------------+    +------------------------+  
 |  MLflow UI           |   |  predict.py loads    |    |  serve.py loads        |  
 |  inspect runs,       |   |  MLflow model and    |    |  MLflow model and      |  
 |  view metrics        |   |  makes sample        |    |  serves REST endpoint  |  
 +----------------------+   |  predictions         |    +------------------------+  
                              +----------------------+                 |              
                                                                      v              
                                                         +------------------------+  
                                                         |  Client sends JSON     |  
                                                         |  request to /predict   |  
                                                         +------------------------+  
``` 

## Workflow steps

1. `run_pipeline.py` starts the workflow.
2. Data is loaded from `src/data/load_data.py`.
3. Features are preprocessed by `src/features/preprocess.py`.
4. Decision Tree model is built by `src/models/train.py`.
5. Model training runs on the preprocessed training set.
6. Predictions are generated on the test set and evaluated by `src/models/evaluate.py`.
7. MLflow logs parameters, metrics, and the trained model artifact.
8. The logged model can be inspected in MLflow UI.
9. `predict.py` loads the same model artifact for sample predictions.
10. `serve.py` loads the model artifact and exposes `/predict` for online inference.

## Key workflow concepts

- `MLflow` is the central tracking and artifact storage system.
- The model artifact is a serialized scikit-learn pipeline.
- `predict.py` demonstrates offline inference.
- `serve.py` demonstrates online inference via REST.
