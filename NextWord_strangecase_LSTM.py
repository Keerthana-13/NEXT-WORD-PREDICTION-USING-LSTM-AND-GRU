#!/usr/bin/env python
# coding: utf-8

# ### Next Word Prediction
# ### Dr.Strange Dataset 

# In[1]:


import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os


# In[5]:


file = open("The_Strange_Case_of_Dr. Jekyll_and_ Mr. Hy _by_Robert_Louis_Stevenson.txt", "r", encoding = "utf8")
lines = []


# In[6]:


for i in file:
    lines.append(i)


# In[7]:


print("The First Line: ", lines[0])
print("The Last Line: ", lines[-1])


# In[8]:


file = open("The_Strange_Case_of_Dr. Jekyll_and_ Mr. Hy _by_Robert_Louis_Stevenson.txt", "rb")
data = file.read()
words = data.split()


# In[9]:


print('Number of words in text file :', len(words))


# ### Cleaning the Data

# In[10]:


data = ""


# In[11]:


for i in lines:
    data = ' '. join(lines)


# In[12]:


data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
data[:360]


# In[13]:


import string


# In[14]:


translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
new_data = data.translate(translator)


# In[15]:


new_data[:500]


# In[16]:


z = []


# In[17]:


for i in data.split():
    if i not in z:
        z.append(i)


# In[18]:


data = ' '.join(z)
data[:500]


# ### Tokenization

# In[19]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])


# In[20]:


# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))


# In[21]:


sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data


# In[22]:


tokenizer.word_index


# In[23]:


vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


# In[24]:


sequences = []


# In[25]:


for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)


# In[26]:


print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]


# In[27]:


X = []
y = []
total_words_dropped = 0


# In[28]:


for i in sequences:
    if len(i) > 1:
        for index in range(1, len(i)):
            X.append(i[:index])
            y.append(i[index])
    else:
        total_words_dropped += 1


# In[29]:


print("Total Single Words Dropped are:", total_words_dropped)


# In[30]:


print("The Data is: ", X[:5])
print("The responses are: ", y[:5])


# In[31]:


X = tf.keras.preprocessing.sequence.pad_sequences(X)


# In[32]:


y = to_categorical(y, num_classes=vocab_size)
y[:5]


# ### Creating the model

# In[33]:


model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))


# In[34]:


model.summary()


# ### Plot the model

# In[35]:


from tensorflow import keras
from keras.utils.vis_utils import plot_model


# In[36]:


keras.utils.plot_model(model, to_file='model.png', show_layer_names=True)


# In[37]:


#Callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard


# In[38]:


checkpoint = ModelCheckpoint("weights_2.h5", monitor='loss', verbose=1,
save_best_only=True, mode='auto')


# In[39]:


reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)


# In[40]:


logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)


# ### Call Backs

# In[41]:


model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001),metrics="accuracy")


# In[43]:


model.evaluate(X,y)


# In[44]:


model = model.fit(X, y, epochs=50, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])


# In[45]:


import matplotlib.pyplot as plt
plt.plot(model.history['accuracy'])
plt.plot(model.history['loss'])
plt.title('model accuracy and loss')
plt.ylabel('accuracy and loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()


# In[44]:


model.save_weights("C:\\Users\\deepak\\Downloads\\Set 3\\weights_2.h5")


# In[45]:


vocab_array = np.array(list(tokenizer.word_index.keys()))


# In[46]:


vocab_array


# In[47]:


# Importing the Libraries
from tensorflow.keras.models import load_model
import numpy as np
import pickle


# In[48]:


# Load the model and tokenizer
model.load_weights("C:\\Users\\deepak\\Downloads\\Set 3\\weights_2.h5")
tokenizer = pickle.load(open('C:\\Users\\deepak\\Downloads\\Set 3\\tokenizer1.pkl', 'rb'))


# In[49]:


def make_prediction(text, n_words):
    for i in range(n_words):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.squeeze(np.argmax(model.predict(text_padded), axis=-1))
        prediction = str(vocab_array[prediction - 1])
        print(vocab_array[np.argsort(model.predict(text_padded)) - 1].ravel()[:-3])
        text += " " + prediction
    return text


# In[50]:


make_prediction("good",5)


# ### Observation:

# We are able to develop a decent next word prediction model and are able to get a declining loss and an overall
# decent performance.

# In[ ]:




