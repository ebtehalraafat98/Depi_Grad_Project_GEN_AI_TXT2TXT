

## 📦 SafeTensor MLflow Model Logger

This project provides a custom MLflow pipeline for logging and packaging a TensorFlow model using the `safetensors` format, HuggingFace tokenizers, and structured metadata. It enables reproducible training, artifact tracking, and deployment-ready model serving via MLflow’s `pyfunc` interface.

---

### 🚀 What It Does

- Wraps a simple TensorFlow model in a custom `mlflow.pyfunc.PythonModel` class  
- Loads model weights from `safetensors` format  
- Logs model architecture, tokenizer files, parameters, and metrics  
- Defines input/output schema for deployment  
- Packages everything into a versioned MLflow model for future loading and inference

---

### 🧠 Technologies Used

- [MLflow](https://mlflow.org/) for experiment tracking and model packaging  
- [TensorFlow](https://www.tensorflow.org/) for model architecture  
- [safetensors](https://github.com/huggingface/safetensors) for secure weight storage  
- [Transformers](https://huggingface.co/docs/transformers/index) for tokenizer integration  
- Python 3.9, Pandas, Cloudpickle

---

### 📂 Project Structure

```
/chat2
  /model                 # Contains safetensors model files
  /token                 # Contains tokenizer files
  parameters-2.json      # Training parameters
  model_logger.py        # Main MLflow logging script
```

---

### 🛠️ How to Use

1. **Prepare your model files**  
   Save your trained model weights in safetensors format inside `/model`.

2. **Prepare tokenizer files**  
   Save HuggingFace tokenizer files inside `/token`.

3. **Define parameters**  
   Create a `parameters-2.json` file with training metadata (e.g., epochs, learning rate).

4. **Run the logger script**  
   Execute `model_logger.py` to start an MLflow run and log all components.

5. **Load the model later**  
   ```python
   model_uri = f"runs:/{run.info.run_id}/model"
   loaded_model = mlflow.pyfunc.load_model(model_uri)
   ```

---

### 📄 Example Parameters File (`parameters-2.json`)

```json
{
  "num_epochs": 10,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

---

### 📌 Notes

- The model is logged with a custom signature and optional input example.  
- You can extend the `SafeTensorModel` class to load weights from `.safetensors` directly using `load_file()` if needed.  
- The environment is packaged using a custom `conda_env` for reproducibility.


