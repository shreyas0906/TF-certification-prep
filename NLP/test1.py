import tensorflow as tf
# from  tf.keras.layers import TextVectorization
from tensorflow.keras import layers

import pandas as pd
import random
from sklearn.model_selection import train_test_split
import sys

print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# print(train_df.head())
train_df_shuffled = train_df.sample(frac=1, random_state=42)

# print(train_df['text'][0])
# print(train_df_shuffled['text'].head())

# print(train_df.target.value_counts())

random_index = random.randint(0, len(train_df) - 5)

# for row in train_df_shuffled[['text', 'target']][random_index : random_index + 5].itertuples():
#     _, text, target = row
#     print(f"Target: {target}", "(real disaster)" if target > 0 else "(not a real disaster)")
#     print(f"Text: \n{text}\n")
#     print("---\n")

train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled['text'].to_numpy(),
                                                                            train_df_shuffled['target'].to_numpy(),
                                                                            test_size=0.1,
                                                                            random_state=42)

print(f"train sentences: {len(train_sentences)} train_labels: {len(train_labels)}")

max_vocab_length = 10000
max_length = 15
# This is defaults.
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_vocab_length,
                                                    # None, # how many different words in the vocabulary, if None, there is no cap on the vocabulary.
                                                    standardize="lower_and_strip_punctuation",
                                                    split='whitespace',
                                                    ngrams=None,
                                                    output_mode="int",  # assign a number(int) to each token
                                                    output_sequence_length=max_length,
                                                    # pad_to_max_tokens=True,
                                                    )

# find the average number of tokens in the training set.
text_vectorizer.adapt(train_sentences)
sample_sentence = "There's a flood in my street!"

print(text_vectorizer([sample_sentence]))

words_in_vocab = text_vectorizer.get_vocabulary()
top_5 = words_in_vocab[:5]
bottom_5 = words_in_vocab[-5:]
print(type(words_in_vocab))
print(f"vocab size: {len(words_in_vocab)}")
print(f"top 5: {top_5}")
print(f"bottom 5: {bottom_5}")

embedding = layers.Embedding(input_dim=max_vocab_length,
                             output_dim=128,
                             input_length=max_length)

random_sentence = random.choice(train_sentences)
print(f"random_sentence: {random_sentence}")
sample_embed = embedding(text_vectorizer([random_sentence]))
print(f"sample_embed: {sample_embed} shape: {sample_embed.shape}")

########################################
# Model 0: Naive bayes with tf-idf encoder
########################################


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create tokenization and modelling pipeline
model_0 = Pipeline([
    ("tfidf", TfidfVectorizer()),  # convert words to numbers using tfidf
    ("clf", MultinomialNB())  # model the text
])

# Fit the pipeline to the training data
model_0.fit(train_sentences, train_labels)
print(model_0)

baseline_score = model_0.score(val_sentences, val_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score * 100:.2f}%")

###########################################
# Model 1: Feed forward dense layer
##########################################
# Build model with the Functional API
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")  # inputs are 1-dimensional strings |||||||| shape=(None, )
x = text_vectorizer(inputs)  # turn the input text into numbers
x = embedding(x)  # create an embedding of the numerized numbers
x = layers.GlobalAveragePooling1D()(
    x)  # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
outputs = layers.Dense(1, activation="sigmoid")(
    x)  # create the output layer, want binary outputs so use sigmoid activation
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")  # construct the model

# Compile model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of the model
model_1.summary()

# Fit the model
model_1_history = model_1.fit(train_sentences,
                              # input sentences can be a list of strings due to text preprocessing layer built-in model
                              train_labels,
                              epochs=5,
                              # batch_size=32,
                              validation_data=(val_sentences, val_labels))
# callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
#                                        experiment_name="simple_dense_model")])

model_1.evaluate(val_sentences, val_labels)

# Make predictions (these come back in the form of probabilities)
model_1_pred_probs = model_1.predict(val_sentences)
print(model_1_pred_probs[:10])  # only print out the first 10 prediction probabilities

# Turn prediction probabilities into single-dimension tensor of floats
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))  # squeeze removes single dimensions
print(model_1_preds[:20])
embed_weights = model_1.get_layer("embedding").get_weights()[0]
print(embed_weights)
# print(embed_weights[0].shape)

import io

# Create output writers
out_v = io.open("embedding_vectors.tsv", "w", encoding="utf-8")
out_m = io.open("embedding_metadata.tsv", "w", encoding="utf-8")

# Write embedding vectors and words to file
for num, word in enumerate(words_in_vocab):
    if num == 0:
        continue  # skip padding token
    vec = embed_weights[num]
    out_m.write(word + "\n")  # write words to file
    out_v.write("\t".join([str(x) for x in vec]) + "\n")  # write corresponding word vector to file
out_v.close()
out_m.close()


##################################################################
# model 2: LSTM model.
##################################################################
model_2_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_2")


# Create LSTM model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_2_embedding(x)
print(x.shape)
# x = layers.LSTM(64, return_sequences=True)(x) # return vector for each word in the Tweet (you can stack RNN cells as long as return_sequences=True)
x = layers.LSTM(64)(x) # return vector for whole sequence
print(x.shape)
# x = layers.Dense(64, activation="relu")(x) # optional dense layer on top of output of LSTM cell
outputs = layers.Dense(1, activation="sigmoid")(x)
model_2 = tf.keras.Model(inputs, outputs, name="model_2_LSTM")

print(model_2.summary())
model_2.compile(loss='binary_crossentropy',
                optimizer="adam",
                metrics=["accuracy"])
model_2.fit(train_sentences, train_labels,
            validation_data=(val_sentences, val_labels))


##################################################################
# model 3: GRU model.
##################################################################
model_2_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_2")


# Create GRU model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_2_embedding(x)
x = layers.GRU(64)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_3 = tf.keras.Model(inputs, outputs, name="model_3_GRU")

#Compile GRU
model_3.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_3.summary()

#Fitting the model.
model_3_history = model_3.fit(train_sentences, train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))

model_3_pred_probs = model_3.predict(val_sentences)
model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
print(f"model_3 probs: {model_3_preds[:10]}")

#################################################################
# model 4: Using bi-directional LSTM
#################################################################
model_4_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_4")

# Build a Bidirectional RNN in TensorFlow
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_4_embedding(x)
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x) # stacking RNN layers requires return_sequences=True
x = layers.Bidirectional(layers.LSTM(64))(x) # bidirectional goes both ways so has double the parameters of a regular LSTM layer
outputs = layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")

# Compile
model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Summary
print(model_4.summary())

# Fit the model (takes longer because of the bidirectional layers)
model_4_history = model_4.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))

model_4_pred_probs = model_4.predict(val_sentences)

# Convert prediction probabilities to labels
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
print(f"model_4_preds: {model_4_preds[:10]}")


##############################################################
# model 5: Using Conv1D model.
##############################################################

model_5_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_5")

# Create 1-dimensional convolutional layer to model sequences
# from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_5_embedding(x)
x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
x = layers.GlobalMaxPool1D()(x)
# x = layers.Dense(64, activation="relu")(x) # optional dense layer
outputs = layers.Dense(1, activation="sigmoid")(x)
model_5 = tf.keras.Model(inputs, outputs, name="model_5_Conv1D")

# Compile Conv1D model
model_5.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of our 1D convolution model
model_5.summary()

model_5.fit(train_sentences, train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels))

##########################################################
# model 6: Using pre-trained embeddings from Tensorflow hub
##########################################################
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed_samples = embed(["I love tensorflow", "When you call the universal sentence encoder, it returns the embeddings."])
# print(embed_samples[0])
# print(embed_samples[0].shape)


sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                      input_shape=[],
                                      dtype=tf.string,
                                      trainable=False,
                                      name='USE_embed_layer')



# Create model using the Sequential API
model_6 = tf.keras.Sequential([
  sentence_encoder_layer, # take in sentences and then encode them into an embedding
  layers.Dense(64, activation="relu"),
  layers.Dense(1, activation="sigmoid")
], name="model_6_USE")

# Compile model
model_6.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_6.summary()

model_6.fit(train_sentences, train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels))

train_10 = train_df[["text", 'target']].sample(frac=0.1)
print(train_10.shape)