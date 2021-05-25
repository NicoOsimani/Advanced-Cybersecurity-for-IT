__author__ = 'Daniele Marzetti'

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, concatenate, Dropout, Dense, Embedding, LSTM, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

max_epoch = 10
batch_size = 512
EMBEDDING_DIMENSION = 128
NUM_CONV_FILTERS = 60
max_features = 38         #?

path = "/content/drive/MyDrive/Cyber Security/Bilbo"

def create_model(MAX_STRING_LENGTH, MAX_INDEX):
    net = {}
    net['input'] = Input((MAX_STRING_LENGTH, ), dtype='int32', name='input')
    
    ########################
    #          CNN         #
    ########################
    
    net['embeddingCNN'] = Embedding(output_dim=EMBEDDING_DIMENSION,
                                    input_dim=MAX_INDEX,
                                    input_length=MAX_STRING_LENGTH,
                                    name='embeddingCNN')(net['input'])
    
    # Parallel Convolutional Layer
    
    net['conv2'] = Conv1D(NUM_CONV_FILTERS, 2, name='conv2')(net['embeddingCNN'])
    
    net['conv3'] = Conv1D(NUM_CONV_FILTERS, 3, name='conv3')(net['embeddingCNN'])
    
    net['conv4'] = Conv1D(NUM_CONV_FILTERS, 4, name='conv4')(net['embeddingCNN'])
    
    net['conv5'] = Conv1D(NUM_CONV_FILTERS, 5, name='conv5')(net['embeddingCNN'])
    
    net['conv6'] = Conv1D(NUM_CONV_FILTERS, 6, name='conv6')(net['embeddingCNN'])
    
    # Global max pooling
    
    net['pool2'] = GlobalMaxPool1D(name='pool2')(net['conv2'])
    
    net['pool3'] = GlobalMaxPool1D(name='pool3')(net['conv3'])
    
    net['pool4'] = GlobalMaxPool1D(name='pool4')(net['conv4'])
    
    net['pool5'] = GlobalMaxPool1D(name='pool5')(net['conv5'])
    
    net['pool6'] = GlobalMaxPool1D(name='pool6')(net['conv6'])
    
    net['concatcnn'] = concatenate([net['pool2'], net['pool3'], net['pool4'
                                   ], net['pool5'], net['pool6']], axis=1,
                                   name='concatcnn')
    
    net['dropoutcnnmid'] = Dropout(0.5, name='dropoutcnnmid')(net['concatcnn'])
    
    net['densecnn'] = Dense(NUM_CONV_FILTERS, activation='relu', name='densecnn')(net['dropoutcnnmid'])
    
    net['dropoutcnn'] = Dropout(0.5, name='dropoutcnn')(net['densecnn'])
    
    ########################
    #         LSTM         #
    ########################
    
    net['embeddingLSTM'] = Embedding(output_dim=max_features,
                                     input_dim=256,
                                     input_length=MAX_STRING_LENGTH,
                                     name='embeddingLSTM')(net['input'])
    
    net['lstm'] = LSTM(256, name='lstm')(net['embeddingLSTM'])
    
    net['dropoutlstm'] = Dropout(0.5, name='dropoutlstm')(net['lstm'])
    
    ########################
    #    Combine - ANN     #
    ########################
    
    net['concat'] = concatenate([net['dropoutcnn'], net['dropoutlstm']], axis=-1, name='concat')
    
    net['dropoutsemifinal'] = Dropout(0.5, name='dropoutsemifinal')(net['concat'])
    
    net['extradense'] = Dense(100, activation='relu', name='extradense')(net['dropoutsemifinal'])
    
    net['dropoutfinal'] = Dropout(0.5, name='dropoutfinal')(net['extradense'])
    
    net['output'] = Dense(1, activation='sigmoid', name='output')(net['dropoutfinal'])
    
    model = Model(net['input'], net['output'])
    return model

def train_eval_test(model, dataset, label):
    earlystop = EarlyStopping(monitor='loss', patience=3)
    best_save = ModelCheckpoint('bestmodel.hdf5', save_best_only=True, save_weights_only= False, monitor='val_loss', mode='min')
    model.compile(optimizer='ADAM', loss= BinaryCrossentropy(), metrics=[BinaryAccuracy(), AUC(), Precision(), Recall()])
    model.summary()
    history = model.fit(x=dataset, y=label.to_numpy(), batch_size=batch_size, epochs=max_epoch, callbacks=[earlystop, best_save], validation_split=0.2)

    fig1 = plt.figure(1)
    plt.title('Loss')
    plt.plot(history.history["val_loss"], 'r', label='Validation Loss')
    plt.plot(history.history["loss"], 'b', label='Training Loss')
    plt.legend(loc="upper right")
    #x = list(range(len(loss_train)+1, 1))
    plt.grid(True)
    fig1.savefig(path + "/bilbo_loss.png")
    plt.show()
    plt.close(fig1)
    
    fig2 = plt.figure(2)
    plt.title('Accuracy')
    plt.plot(history.history["val_binary_accuracy"], 'r', label='Validation Accuracy')
    plt.plot(history.history["binary_accuracy"], 'b', label='Training Accuracy')
    plt.legend(loc="lower right")
    plt.grid(True)
    fig2.savefig(path + "/bilbo_accuracy.png")
    plt.show()
    plt.close(fig2)
    
    best_model = load_model('bestmodel.hdf5')
    best_model.evaluate(x=dataset, y=label.to_numpy(), batch_size=batch_size)
    predicted = best_model.predict(x=dataset, batch_size=batch_size)



def use_tpu():
    print("Tensorflow version " + tf.__version__)

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.TPUStrategy(tpu)