from flask import Flask, request, render_template
import joblib
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel ,TFAutoModel
import xgboost

xgb = joblib.load('model3/xgb_model.joblib')
model_name = 'bert-large-cased'

minMaxDiff = 34460
minVal = 0
currentIndex = 0
# ['laughingTime','jokeLength','sentiment','sentiment_prob']
df = pd.DataFrame(columns=['text','laughingTime','jokeLength','sentiment','sentiment_prob','rank'])

tokenizer = BertTokenizer.from_pretrained(model_name)

model2 = TFBertModel.from_pretrained(model_name)

base_model = TFAutoModel.from_pretrained('bert-large-uncased')
base_model.trainable = False

input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name='attention_mask')

x = base_model({"input_ids": input_ids, "attention_mask": attention_mask})[1]
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs={"input_ids": input_ids, "attention_mask": attention_mask}, outputs=outputs)
model.load_weights('model/model3_large_minmax_extradata/my_model')
tokenizer1 = BertTokenizer.from_pretrained('bert-large-uncased')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    global df,currentIndex

    sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    text = request.form['text']
    currentIndex = addRowToDF([{'text':text,'laughingTime':'','jokeLength':len(text.split()),'sentiment':'','sentiment_prob':'','rank':''}])
    duration = get_laugh_Duration(df['text'][currentIndex])
    df['laughingTime'][currentIndex] = duration[0]
    sentiment = getSentiment(df['text'][currentIndex])
    df['sentiment'][currentIndex] = sentiment[0]
    df['sentiment_prob'][currentIndex] = sentiment[1]
    df['rank'][currentIndex] = get_rank(pd.DataFrame(df.iloc[currentIndex].to_dict(), index=[0])[['laughingTime','jokeLength','sentiment','sentiment_prob']])[0]
    df['sentiment'][currentIndex] = sentiment_map[df['sentiment'][currentIndex]]
    return df.iloc[currentIndex].to_dict()


def get_rank(input):
    global xgb
    score = xgb.predict(input)
    return score

def getSentiment(text):
    global tokenizer,model2
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding='max_length', return_tensors='tf')
    output = model2(tokens)
    pooled_output = output[1]
    sentiment_logits = tf.keras.layers.Dense(3, activation='softmax')(pooled_output)
    sentiment_probabilities = tf.nn.softmax(sentiment_logits, axis=1).numpy().squeeze()
    sentiment_label = tf.argmax(sentiment_probabilities).numpy().item()

    # Map the sentiment label to the corresponding sentiment
    # sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    return [sentiment_label, sentiment_probabilities[sentiment_label]]

# {'laughingTime':,'jokeLength':,'sentiment':,'sentiment_prob':}
def addRowToDF(inp):
    global df
    new_row = pd.DataFrame(inp)
    df = pd.concat([df, new_row], ignore_index=True)
    new_row_index = df.index[-1]
    return new_row_index



def dataPreProcess(texts):
    # Load the BERT tokenizer
    global tokenizer1

    # Define the maximum sequence length for padding/truncating
    max_length = 512

    # Tokenize the list of strings and convert them to input IDs, attention masks, and token type IDs
    input_ids = []
    attention_masks = []

    for text in texts:
        # Tokenize the text and add the special [CLS] and [SEP] tokens
        encoded_dict = tokenizer1.encode_plus(
                            text,                      # Text to encode
                            add_special_tokens = True, # Add [CLS] and [SEP] tokens
                            max_length = max_length,   # Pad/truncate to a maximum length
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Generate attention masks
                            return_token_type_ids = False,   # Do not generate token type IDs
                            truncation=True,
                            )
        
        # Add the encoded sequence and attention mask to the lists
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists to tensors
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_masks = tf.convert_to_tensor(attention_masks, dtype=tf.int32)
    
    # Return a tuple of input IDs and attention masks
    return input_ids, attention_masks

def get_laugh_Duration(text):
    global minMaxDiff,minVal,model
    data = dataPreProcess([text])
    inputs = {'input_ids': data[0], 'attention_mask': data[1]}
    preds = model.predict(inputs).tolist()
    val_preds = [((item[0]*minMaxDiff)+minVal) for item in list(preds)]
    return val_preds



if __name__ == '__main__':
    app.run(debug=True, port=8080)
