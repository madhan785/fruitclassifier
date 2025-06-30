import os
import tensorflow as tf
from preprocessing import DataLoader,call_back
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Activation,GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import json
import config

loader=DataLoader()
loader.load_data()
cls=loader.get_classes()
call_backs=call_back()

#base model from the pretrained model 
base_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=config.INPUT_SHAPE)
base_model.trainable=False

#Model
model=Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(123,kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation(config.ACTIVATION_FUNCTION),
        Dropout(0.3),
        Dense(64,kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Activation(config.ACTIVATION_FUNCTION),
        Dropout(0.3),
        Dense(len(cls),activation='softmax')
    ]
)

model.compile(optimizer=config.OPTIMIZER,loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
history=model.fit(loader.train_data,epochs=config.EPOCHS,validation_data=loader.test_data,batch_size=config.BATCH_SIZE,callbacks=call_backs.get_callbacks())

#save model
model.save(config.MODEL_SAVE_PATH)
print('model saved succesfully')

#save history
with open(config.MODEL_HISTORY_PATH,'w') as file:
    json.dump(history.history,file)

#save model architecture
model_json = model.to_json()
with open(config.MODEL_ARCHITECTURE_PATH, "w") as json_file:
    json.dump(model_json, json_file)