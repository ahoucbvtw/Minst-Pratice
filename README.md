## Minst-Pratice
*** 
### List

[Minst_Number Step Description](https://github.com/ahoucbvtw/Minst-Pratice#minst_number-step-description)

[Minst_Fashion Step Description](https://github.com/ahoucbvtw/Minst-Pratice#minst_fashion-step-description)

***
### Minst_Number Step Description
1. Import Minst_Number DataSet
```
from tensorflow.keras.datasets.mnist import load_data

(traindata, trainanswer), (testdata, testanswer) = load_data()
```

2. Build Dense layer
```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_5 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
```

3. The answer's OneHotEncoding
```
from tensorflow.keras.utils import to_categorical

trainanswer_cat = to_categorical(trainanswer, num_classes = 10)
testanswer_cat = to_categorical(testanswer, num_classes = 10)
```

4. Training and Testing pictures preprocess(Normalize)
```
traindata_norm = traindata.reshape(-1, 784) / 255
testdata_norm = testdata.reshape(-1, 784) / 255
```

5. Training
```
Epoch 1/100
270/270 - 1s - loss: 0.4380 - accuracy: 0.8785 - val_loss: 0.1953 - val_accuracy: 0.9432
Epoch 2/100
270/270 - 1s - loss: 0.2015 - accuracy: 0.9426 - val_loss: 0.1379 - val_accuracy: 0.9623
Epoch 3/100
270/270 - 1s - loss: 0.1487 - accuracy: 0.9577 - val_loss: 0.1174 - val_accuracy: 0.9660
Epoch 4/100
270/270 - 1s - loss: 0.1185 - accuracy: 0.9661 - val_loss: 0.0995 - val_accuracy: 0.9718
Epoch 5/100
270/270 - 1s - loss: 0.0978 - accuracy: 0.9719 - val_loss: 0.0945 - val_accuracy: 0.9720
Epoch 6/100
270/270 - 1s - loss: 0.0825 - accuracy: 0.9759 - val_loss: 0.0857 - val_accuracy: 0.9747
Epoch 7/100
270/270 - 1s - loss: 0.0693 - accuracy: 0.9807 - val_loss: 0.0828 - val_accuracy: 0.9760
Epoch 8/100
270/270 - 1s - loss: 0.0603 - accuracy: 0.9830 - val_loss: 0.0784 - val_accuracy: 0.9762
Epoch 9/100
270/270 - 1s - loss: 0.0521 - accuracy: 0.9851 - val_loss: 0.0794 - val_accuracy: 0.9772
Epoch 10/100
270/270 - 1s - loss: 0.0455 - accuracy: 0.9876 - val_loss: 0.0751 - val_accuracy: 0.9790
Epoch 11/100
270/270 - 1s - loss: 0.0394 - accuracy: 0.9896 - val_loss: 0.0790 - val_accuracy: 0.9768
Epoch 12/100
270/270 - 1s - loss: 0.0340 - accuracy: 0.9911 - val_loss: 0.0731 - val_accuracy: 0.9793
Epoch 13/100
270/270 - 1s - loss: 0.0303 - accuracy: 0.9920 - val_loss: 0.0753 - val_accuracy: 0.9798
Epoch 14/100
270/270 - 1s - loss: 0.0267 - accuracy: 0.9939 - val_loss: 0.0718 - val_accuracy: 0.9798
Epoch 15/100
270/270 - 1s - loss: 0.0235 - accuracy: 0.9944 - val_loss: 0.0730 - val_accuracy: 0.9805
Epoch 16/100
270/270 - 1s - loss: 0.0201 - accuracy: 0.9960 - val_loss: 0.0739 - val_accuracy: 0.9792
Epoch 17/100
270/270 - 1s - loss: 0.0179 - accuracy: 0.9960 - val_loss: 0.0751 - val_accuracy: 0.9795
Epoch 18/100
270/270 - 1s - loss: 0.0155 - accuracy: 0.9971 - val_loss: 0.0732 - val_accuracy: 0.9805
Epoch 19/100
270/270 - 1s - loss: 0.0136 - accuracy: 0.9976 - val_loss: 0.0761 - val_accuracy: 0.9807
```

6. Make a Confusion_Matrix to see the model's accurancy

![ ](https://raw.githubusercontent.com/ahoucbvtw/Minst-Pratice/main/Minst_Number/Picture/Confusion_Matrix.jpg)

7. Print those wrong predict's pictures with real and predict answer

![ ](https://raw.githubusercontent.com/ahoucbvtw/Minst-Pratice/main/Minst_Number/Picture/Wrong%20number.jpg)

[Top](https://github.com/ahoucbvtw/Minst-Pratice#minst-pratice)

***
### Minst_Fashion Step Description

1. Import Minst_Fashion DataSet
```
from tensorflow.keras.datasets.fashion_mnist import load_data

(traindata, trainanswer), (testdata, testanswer) = load_data()
```

2. 2. Build Dense layer
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
```

3. Beacause of I use **SparseCategoricalCrossentropy**, the answers do not OneHotEncoding
```
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss = SparseCategoricalCrossentropy(),
              optimizer="adam",
              metrics = ["accuracy"])
```

4. Training and Testing pictures preprocess(Normalize)
```
traindata_norm = traindata.reshape(-1, 784) / 255
testdata_norm = testdata.reshape(-1, 784) / 255
```

5. Training
```
Epoch 1/100
270/270 - 2s - loss: 0.6054 - accuracy: 0.7941 - val_loss: 0.4695 - val_accuracy: 0.8267
Epoch 2/100
270/270 - 1s - loss: 0.4247 - accuracy: 0.8548 - val_loss: 0.4176 - val_accuracy: 0.8532
Epoch 3/100
270/270 - 1s - loss: 0.3852 - accuracy: 0.8657 - val_loss: 0.3771 - val_accuracy: 0.8638
Epoch 4/100
270/270 - 1s - loss: 0.3581 - accuracy: 0.8736 - val_loss: 0.3696 - val_accuracy: 0.8712
Epoch 5/100
270/270 - 1s - loss: 0.3386 - accuracy: 0.8791 - val_loss: 0.3722 - val_accuracy: 0.8678
Epoch 6/100
270/270 - 1s - loss: 0.3204 - accuracy: 0.8859 - val_loss: 0.3466 - val_accuracy: 0.8753
Epoch 7/100
270/270 - 1s - loss: 0.3071 - accuracy: 0.8896 - val_loss: 0.3381 - val_accuracy: 0.8777
Epoch 8/100
270/270 - 1s - loss: 0.2951 - accuracy: 0.8940 - val_loss: 0.3332 - val_accuracy: 0.8808
Epoch 9/100
270/270 - 1s - loss: 0.2846 - accuracy: 0.8967 - val_loss: 0.3423 - val_accuracy: 0.8780
Epoch 10/100
270/270 - 1s - loss: 0.2760 - accuracy: 0.9002 - val_loss: 0.3402 - val_accuracy: 0.8777
Epoch 11/100
270/270 - 1s - loss: 0.2658 - accuracy: 0.9035 - val_loss: 0.3288 - val_accuracy: 0.8817
Epoch 12/100
270/270 - 1s - loss: 0.2601 - accuracy: 0.9060 - val_loss: 0.3156 - val_accuracy: 0.8855
Epoch 13/100
270/270 - 1s - loss: 0.2546 - accuracy: 0.9069 - val_loss: 0.3187 - val_accuracy: 0.8870
Epoch 14/100
270/270 - 1s - loss: 0.2490 - accuracy: 0.9102 - val_loss: 0.3343 - val_accuracy: 0.8765
Epoch 15/100
270/270 - 1s - loss: 0.2402 - accuracy: 0.9124 - val_loss: 0.3054 - val_accuracy: 0.8903
Epoch 16/100
270/270 - 1s - loss: 0.2333 - accuracy: 0.9145 - val_loss: 0.3131 - val_accuracy: 0.8875
Epoch 17/100
270/270 - 1s - loss: 0.2258 - accuracy: 0.9177 - val_loss: 0.3154 - val_accuracy: 0.8862
Epoch 18/100
270/270 - 1s - loss: 0.2252 - accuracy: 0.9178 - val_loss: 0.3147 - val_accuracy: 0.8828
Epoch 19/100
270/270 - 1s - loss: 0.2149 - accuracy: 0.9213 - val_loss: 0.3183 - val_accuracy: 0.8832
Epoch 20/100
270/270 - 1s - loss: 0.2108 - accuracy: 0.9233 - val_loss: 0.3215 - val_accuracy: 0.8848
```

6. Make a Confusion_Matrix to see the model's accurancy

![ ](https://raw.githubusercontent.com/ahoucbvtw/Minst-Pratice/main/Minst_Fashion/Picture/Confusion_Matrix.jpg) 

[Top](https://github.com/ahoucbvtw/Minst-Pratice#minst-pratice)
