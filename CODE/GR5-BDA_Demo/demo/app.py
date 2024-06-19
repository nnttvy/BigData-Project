from flask import Flask, render_template, request
import string, re, pandas as pd, numpy as np, html
import pickle, joblib
from gensim.models import Word2Vec
import matplotlib.pyplot as plt, seaborn as sns
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
vectorizer = joblib.load('models/count_vectorizer.pkl')
lr_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
w2v_model = Word2Vec.load('models/word2vec_model.bin')
model = joblib.load('models/naive_bayes_model.pkl')
lr_model = pickle.load(open('models/logistic_regression_model.pkl', 'rb'))
lstm_model = load_model('models/lstm_model.h5')

label_dict = {
    0: 'Animals',
    1: 'Education',
    2: 'Film & Animation',
    3: 'Food',
    4: 'Gaming',
    5: 'History & Documentary',
    6: 'Music',
    7: 'News & Politics',
    8: 'NonProfit & Activism',
    9: 'Science & Technology',
    10: 'Sports',
    11: 'Travel & Events'
}

rcm_path = r'rs/rcm-vid-sample.csv'
recommend_data = pd.read_csv(rcm_path)
recommend_df = recommend_data[['clean_title', 'Topic']].copy()
recommend_df.dropna(inplace=True)
topic = recommend_df['Topic']

def is_english(word):
    for char in word:
        if not ('a' <= char.lower() <= 'z'):
            return False
    return True

def remove_non_english(text):
    words = text.split()
    output = []
    current_word = ''
    for word in words:
        if is_english(word):
            current_word += word + ' '
        elif current_word:
            output.append(current_word.strip())
            current_word = ''
    if current_word:
        output.append(current_word.strip())
    return ' '.join(output)

def processing(text):
    text = re.sub(r'\&\w*;', '', text)
    text = re.sub('@[^\s]+','', text)
    text = re.sub(r'\$\w*', '', text)
    text = text.lower()
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    text = re.sub(r'#\w*', '', text)
    text = re.sub(r'['+string.punctuation+']+', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s\s+', ' ', text)
    text = text.lstrip(' ')
    text = ''.join(c for c in text if c <= '\uFFFF')
    return text

def get_vector(word_list, model):
    vec = np.zeros(model.vector_size).reshape((1, model.vector_size))
    count = 0.
    for word in word_list:
        if word in model.wv.key_to_index:
            vec += model.wv.get_vector(word).reshape((1, model.vector_size))
            count += 1.
    if count != 0:
        vec /= count
    return vec

def predict(text):
    processed_text = remove_non_english(text)
    processed_text = processing(processed_text)
    # Naive Bayes
    vectorized_text = vectorizer.transform([processed_text])
    predicted_label_nb = model.predict(vectorized_text)[0]
    decoded_label_nb = label_dict[predicted_label_nb]
    accuracy_nb = round(model.predict_proba(vectorized_text).max() * 100, 2)
    # Logistic Regression
    vectorized_text_lr = lr_vectorizer.transform([processed_text])
    decoded_label_lr = lr_model.predict(vectorized_text_lr)[0]
    accuracy_lr = round(lr_model.predict_proba(vectorized_text_lr).max() * 100, 2)
    # LSTM
    word_vectors = get_vector(processed_text, w2v_model)
    word_vectors = word_vectors.reshape(-1, 1, 100)
    prediction_lstm = lstm_model.predict(word_vectors)
    predicted_label_lstm = np.argmax(prediction_lstm)
    decoded_label_lstm = label_dict[predicted_label_lstm]
    accuracy_lstm = round((prediction_lstm.max() * 100), 2)
    return decoded_label_nb, accuracy_nb, decoded_label_lr, accuracy_lr, decoded_label_lstm, accuracy_lstm

def viz(models, accuracy_scores, labels):
    x = np.arange(len(models))
    width = 0.6
    _, ax = plt.subplots(figsize=(7, 4))
    unique_colors = sns.color_palette('Set2', len(np.unique(labels)))
    color_dict = {label: color for label, color in zip(np.unique(labels), unique_colors)}
    bars = []
    for i, (accuracy_score, label) in enumerate(zip(accuracy_scores, labels)):
        color = color_dict[label]
        bar = ax.bar(i, accuracy_score, width, label=label, color=color)
        bars.append(bar[0])
    ax.set_ylim([0, 131])
    ax.set_ylabel('ACCURACY (%)')
    ax.set_title('BAR CHART - COMPARISON OF MODEL ACCURACIES', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    custom_legend = [plt.Rectangle((0, 0), 1, 1, color=color) for color in sns.color_palette('Set2', len(models))]
    ax.legend(custom_legend, np.unique(labels))
    
    def autolabel(rects):
        for _, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    autolabel(bars)
    
    plt.savefig('static/accuracy_comparison.png')
    plt.clf()
    return labels

def check_prediction(label_nb, label_lr, label_lstm):
    labels = [label_nb, label_lr, label_lstm]
    unique_labels = set(labels)
    if len(unique_labels) == 1:
        return unique_labels.pop()
    else:
        label_counts = {label: labels.count(label) for label in unique_labels}
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_labels[0][0]
    
def compute_similarity(title, videos, labels, predicted_label, vectorizer, recommended_title):
    same_label_indices = [i for i, label in enumerate(labels) if label == predicted_label]
    same_label_videos = videos.iloc[same_label_indices].dropna().tolist()
    same_label_parameters = recommended_title.iloc[same_label_indices]
    title_vector = vectorizer.transform([title])
    same_label_vectors = vectorizer.transform(same_label_videos)
    similarities = cosine_similarity(title_vector, same_label_vectors)
    top_indices = np.argsort(similarities.flatten())[-5:]
    top_videos = same_label_parameters.iloc[top_indices].tolist()
    title_mapping = dict(zip(recommend_data['clean_title'], recommend_data['Title']))
    top_videos_with_titles = []
    for clean_title in top_videos:
        clean_title = clean_title.strip('()')
        title_parts = clean_title.split(', ')
        video_title = title_parts[0].strip()
        video_title = html.unescape(video_title)
        top_videos_with_titles.append((title_mapping[video_title], clean_title))
    top_videos_with_titles = [i[0] for i in top_videos_with_titles]
    return top_videos_with_titles

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def backend():
    if request.method == 'POST':
        text = request.form['text']
        decoded_label_nb, accuracy_nb, decoded_label_lr, accuracy_lr, decoded_label_lstm, accuracy_lstm = predict(text)
        predicted_label = check_prediction(decoded_label_nb, decoded_label_lr, decoded_label_lstm)
        models = ['Naive Bayes', 'Logistic Regression', 'LSTM']
        accuracy_scores = [accuracy_nb, accuracy_lr, accuracy_lstm]
        labels = [decoded_label_nb, decoded_label_lr, decoded_label_lstm]
        viz(models, accuracy_scores, labels)
        recommended_videos = compute_similarity(text, recommend_df['clean_title'], topic , predicted_label, vectorizer, recommend_df['clean_title'])
        return render_template('result.html', 
                               prediction_nb=decoded_label_nb, accuracy_nb=accuracy_nb,
                               prediction_lr=decoded_label_lr, accuracy_lr=accuracy_lr,
                               prediction_lstm=decoded_label_lstm, accuracy_lstm=accuracy_lstm,
                               labels=labels,
                               recommended_videos=recommended_videos)

if __name__ == '__main__':
    app.run(debug=True)