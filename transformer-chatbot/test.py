import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from utils import load_data, load_tokenizers, create_tokenizers, prepare_data
from layers import CustomSchedule, Transformer, create_masks, loss_function
# from get_reddit_data import get_data
from dataloader import loader
import os
import time
import yaml
import random

print(f"using tensorflow v{tf.__version__}")
print(f"using tensorflow.keras v{tf.keras.__version__}")


class Chatbot(object):

  def __init__(self, config_path):

    with open(os.path.join(config_path,"config.yml")) as cf:
      config = yaml.load(cf, Loader=yaml.FullLoader)

    self.num_layers = config["num_layers"]
    self.d_model = config["d_model"]
    self.dff = config["dff"]
    self.num_heads = config["num_heads"]
    self.dropout_rate = config["dropout_rate"]
    self.max_length = config["max_length"]
    self.target_vocab_size = config["target_vocab_size"]
    self.checkpoint = config["checkpoint"]
    self.max_checkpoint = config["max_checkpoint"]
    self.custom_checkpoint = config["custom_checkpoint"]
    
    if config["storage_path"] != None:
      self.storage_path = config["storage_path"]
    else:
      self.storage_path = "./"
    
    if config["ckpt_path"] != None:
      self.ckpt_path = config["ckpt_path"]
    else:
      self.ckpt_path = "./"

    if not self.storage_path.endswith("/"):
      self.storage_path += "/"
    
    if not self.ckpt_path.endswith("/"):
      self.ckpt_path += "/"

    self.data_path = f"{self.storage_path}data"
    self.checkpoint_path = f"{self.ckpt_path}checkpoints/train"
    self.tokenizer_path = f"{self.storage_path}tokenizers"
    self.inputs_savepath = f"{self.tokenizer_path}/inputs_token"
    self.outputs_savepath = f"{self.tokenizer_path}/outputs_token"

    
    try:
      self.inputs_tokenizer, self.outputs_tokenizer = load_tokenizers(
        inputs_outputs_savepaths=[self.inputs_savepath, self.outputs_savepath])
    except:
      print("No tokenizers has been created yet, creating new tokenizers...")
      self.inputs_tokenizer, self.outputs_tokenizer = create_tokenizers(
        inputs_outputs=[self.inputs, self.outputs],
        inputs_outputs_savepaths=[self.inputs_savepath, self.outputs_savepath],
        target_vocab_size=self.target_vocab_size)

    self.input_vocab_size = self.inputs_tokenizer.vocab_size + 2
    self.target_vocab_size = self.outputs_tokenizer.vocab_size + 2

    self.learning_rate = CustomSchedule(self.d_model)
    self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    self.transformer = Transformer(
      self.num_layers, self.d_model,
      self.num_heads, self.dff,
      self.input_vocab_size,
      self.target_vocab_size,
      pe_input=self.input_vocab_size,
      pe_target=self.target_vocab_size,
      rate=self.dropout_rate)

    self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                               optimizer=self.optimizer)
    self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=self.max_checkpoint)

    if self.custom_checkpoint:
      self.ckpt.restore(self.custom_checkpoint)
      print(f"Custom checkpoint restored: {self.custom_checkpoint}")
    # if a checkpoint exists, restore the latest checkpoint.
    elif self.ckpt_manager.latest_checkpoint:
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
      print (f"Latest checkpoint restored: {self.ckpt_manager.latest_checkpoint}")

    

  def predict(self, usr_input):
    predicted_sentence, _, _, _ = self.reply(usr_input)
    return predicted_sentence


 
  def evaluate(self, inp_sentence):
    start_token = [self.inputs_tokenizer.vocab_size]
    end_token = [self.inputs_tokenizer.vocab_size + 1]
    
    inp_sentence = start_token + self.inputs_tokenizer.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    decoder_input = [self.outputs_tokenizer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)
      
    for i in range(self.max_length):
      enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
          encoder_input, output)
    
      predictions, attention_weights = self.transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
      
      predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      
      if predicted_id == self.outputs_tokenizer.vocab_size+1:
        return tf.squeeze(output, axis=0), attention_weights
      
      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

  def reply(self, sentence):
    result, attention_weights = self.evaluate(sentence)
    
    predicted_sentence = self.outputs_tokenizer.decode([i for i in result 
                                              if i < self.outputs_tokenizer.vocab_size])

    return predicted_sentence, attention_weights, sentence, result




# if __name__ == "__main__":
# CONFIG_PATH = "."
# chatbot = Chatbot(CONFIG_PATH)
# for i in range(5):
#   usr_input = 'Hi Iam feeling quite down today'
#   predicted_sentence = chatbot.predict(usr_input)
#   print(predicted_sentence)