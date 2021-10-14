import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed_samples = embed(["I love tensorflow", "When you call the universal sentence encoder, it returns the embeddings."])
# print(embed_samples[0])
# print(embed_samples[0].shape)


setence_encode_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                      input_shape=[],
                                      dtype=tf.string,
                                      trainable=False,
                                      name='USE_embed_layer')

