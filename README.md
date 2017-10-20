# human-recognition
Classification algorithm for classification human, on biometric data, using siamese neural networks using Keras.

# requirements
- download dataset and extract it to train_db, test_db
- preprocess.ipynb - creates train and test dataset, saves it in **'dataset/train'** and **'dataset/test'** in **'*.h5'** format
- train.py - trains best model, saves its and train_history in **'models/'** and **'history/'**
- evaluate.ipynb - loads model, makes predictions, saves it and shows Equal Error Rate with graph

# Best result
- **EER = 10%**
- model - **'models/best_model.h5'**
- model structure - **'best_structure'**

# Training
On GeForce GTX 760 CUDA
