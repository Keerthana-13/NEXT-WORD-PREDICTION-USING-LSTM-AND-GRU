#!/usr/bin/env python
# coding: utf-8

# # Next Word Prediction:
# ## Mother Earth

# ### Importing The Required Libraries:

# In[1]:


import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM,GRU,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os


# In[2]:


"""
    Dataset: http://www.gutenberg.org/cache/epub/5200/pg5200.txt
    Remove all the unnecessary data and label it as Metamorphosis-clean.
    The starting and ending lines should be as follows.

"""


file = open("C:/Users/Admin/Downloads/Mother_Earth.txt", "r", encoding = "utf8")
lines = []

for i in file:
    lines.append(i)
    
print("The First Line: ", lines[0])
print("The Last Line: ", lines[-1])


# ### Cleaning the data:

# In[3]:


data = ""

for i in lines:
    data = ' '. join(lines)
    
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
data[:360]


# In[4]:


import string

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
new_data = data.translate(translator)

new_data[:500]


# In[5]:


z = []

for i in data.split():
    if i not in z:
        z.append(i)
        
data = ' '.join(z)
data[:500]


# ### Tokenization:

# In[6]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:10]


# In[7]:


vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


# In[8]:


sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]


# In[9]:


X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])
    
X = np.array(X)
y = np.array(y)


# In[10]:


print("The Data is: ", X[:5])
print("The responses are: ", y[:5])


# In[11]:


y = to_categorical(y, num_classes=vocab_size)
y[:5]


# ### Creating the Model:

# In[12]:


model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(GRU(1000, return_sequences=True))
model.add(GRU(1000))
model.add(Dense(1000, activation="tanh"))
model.add(Dense(vocab_size, activation="softmax"))


# In[13]:


model.summary()


# In[14]:


get_ipython().system('pip install pydot')


# In[15]:


pip install graphviz


# ### Plot The Model:

# In[16]:


import pydot
import graphviz
from tensorflow import keras
from keras.utils.vis_utils import plot_model

keras.utils.plot_model(model, to_file='model.png', show_layer_names=True)


# ### Callbacks:

# In[14]:


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)


# ### Compile The Model:

# In[15]:


model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001),metrics="accuracy")


# In[16]:


model=model.fit(X, y, epochs=50, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])


# In[17]:


import matplotlib.pyplot as plt
plt.plot(model.history['accuracy'])
plt.plot(model.history['loss'])
plt.title('model accuracy and loss')
plt.ylabel('accuracy and loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()


# In[20]:


model.evaluate(X,y)


# In[22]:


model.save_weights("C:/Users/Admin/Downloads/nextword1.h5")
vocab_array = np.array(list(tokenizer.word_index.keys()))

# Importing the Libraries
from tensorflow.keras.models import load_model
import numpy as np
import pickle
# Load the model and tokenizer
model.load_weights("C:/Users/Admin/Downloads/nextword1.h5")
tokenizer = pickle.load(open("C:/Users/Admin/Downloads/tokenizer1.pkl", 'rb'))
def make_prediction(text, n_words):
    for i in range(n_words):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.squeeze(np.argmax(model.predict(text_padded), axis=-1))
        prediction = str(vocab_array[prediction - 1])
        print(vocab_array[np.argsort(model.predict(text_padded)) - 1].ravel()[:-3])
        text += " " + prediction
    return text


# In[23]:


make_prediction("child",5)

