# Detecting AI Generated Faces Among Real Faces

## Dataset Used
```
Total number of images: 1288
Number of "Fake" faces: 700
Number of "Real" faces: 589
```
https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces

## Model

Network: <br />
```
Classifier( 
  (main): Sequential( 
    (0): Conv2d(3, 224, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
    (1): ReLU() 
    (2): Conv2d(224, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
    (3): MaxPool2d(kernel_size=5, stride=1, padding=0, dilation=1, ceil_mode=False)
    (4): ReLU()
    (5): Conv2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False) 
    (6): MaxPool2d(kernel_size=5, stride=1, padding=0, dilation=1, ceil_mode=False)
    (7): ReLU() 
    (8): Flatten(start_dim=1, end_dim=-1)
    (9): Linear(in_features=57600, out_features=2, bias=True)
    (10): Sigmoid() 
  ) 
)
```

Loss: `CrossEntropyLoss()`
<br />
Optimizer: `Adam(learning_rate=0.0001, betas=(0.5, 0.999))`

## Results
Results after training on CUDA with 10 Epochs
```
Epoch 1/10
-------------------------
Loss: 0.5951 Acc: 71.45%

Epoch 2/10
-------------------------
Loss: 0.3914 Acc: 95.73%

Epoch 3/10
-------------------------
Loss: 0.3447 Acc: 98.45%

Epoch 4/10
-------------------------
Loss: 0.3319 Acc: 98.99%

Epoch 5/10
-------------------------
Loss: 0.3266 Acc: 99.07%

Epoch 6/10
-------------------------
Loss: 0.3215 Acc: 99.77%

Epoch 7/10
-------------------------
Loss: 0.3191 Acc: 99.69%

Epoch 8/10
-------------------------
Loss: 0.3174 Acc: 99.84%

Epoch 9/10
-------------------------
Loss: 0.3173 Acc: 99.77%

Epoch 10/10
-------------------------
Loss: 0.3156 Acc: 99.92%
```

### Result examples, visualized
![Results](https://user-images.githubusercontent.com/111753936/204688863-bbd56091-caf0-4c07-a151-b8a6883d92dc.png)


