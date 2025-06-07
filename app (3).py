#!/usr/bin/env python
# coding: utf-8

# # Mounting Google Drive
# 

# In[1]:




# # Import Library
# 

# In[2]:
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras import layers
from keras import ops
import joblib


# # Downloading Dataset

# In[3]:


files.upload()


# In[4]:


get_ipython().system('kaggle datasets download -d arashnic/book-recommendation-dataset -p /content/drive/MyDrive/ML-Project/Datasets')
get_ipython().system('unzip /content/drive/MyDrive/ML-Project/Datasets/book-recommendation-dataset.zip -d /content/drive/MyDrive/ML-Project/Datasets')


# # Loading Dataset
# 

# In[5]:


books = pd.read_csv('/content/drive/MyDrive/ML-Project/Datasets/Books.csv')
ratings = pd.read_csv('/content/drive/MyDrive/ML-Project/Datasets/Ratings.csv')
users = pd.read_csv('/content/drive/MyDrive/ML-Project/Datasets/Users.csv')


# In[6]:


jml_baris, jml_kolom = books.shape
print('jumlah baris', jml_baris)
print('jumlah kolom', jml_kolom)

books.head()


# In[7]:


jml_baris, jml_kolom = ratings.shape
print('jumlah baris', jml_baris)
print('jumlah kolom', jml_kolom)

ratings.head()


# In[8]:


jml_baris, jml_kolom = users.shape # Mendapatkan jumlah baris dan kolom
print('jumlah baris', jml_baris) # Menampilkan jumlah baris
print('jumlah kolom', jml_kolom) # Menampilkan jumlah kolom

users.head() # Menampilkan 5 baris teratas


# # EDA

# In[9]:


books.describe()
ratings.describe()
users.describe()


# In[10]:


books.info()


# In[11]:


# jika langsung menjalankan kode ini maka terjadi error karena ada kesalahan input
# books['Year-Of-Publication'].astype('int')

# Year-Of-Publication yang terjadi kesalahan input
books[(books['Year-Of-Publication'] == 'DK Publishing Inc') | (books['Year-Of-Publication'] == 'Gallimard')]


# In[12]:


books = books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)
books.head()


# In[13]:


# melihat berapa banyak entri dari masing - masing variabel
print("Jumlah nomor ISBN Buku:", len(books['ISBN'].unique()))
print("Jumlah judul buku:", len(books['Book-Title'].unique()))
print('Jumlah penulis buku:', len(books['Book-Author'].unique()))
print('Jumlah Tahun Publikasi:', len(books['Year-Of-Publication'].unique()))
print('Jumlah nama penerbit:', len(books['Publisher'].unique()))


# In[14]:


# Grouping'Book-Author' dan hitung jumlah buku yang ditulis oleh masing-masing penulis
author_counts = books.groupby('Book-Author')['Book-Title'].count()

# Urutkan penulis dalam urutan menurun
sorted_authors = author_counts.sort_values(ascending=False)

# Pilih 10 penulis teratas
top_10_authors = sorted_authors.head(10)

# Plot 10 penulis teratas dan buku yang ditulis oleh penulis kemudian dihitung menggunakan plot batang
plt.figure(figsize=(12, 6))
top_10_authors.plot(kind='bar')
plt.xlabel('Nama Penulis')
plt.ylabel('Jumlah Buku')
plt.title('10 Penulis Teratas Berdasarkan Jumlah Buku')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[15]:


ratings.info()


# In[16]:


print('Jumlah User-ID:', len(ratings['User-ID'].unique()))
print('Jumlah buku berdasarkan ISBN:', len(ratings['ISBN'].unique()))

print('Jumlah rating buku:')
sorted_ratings = ratings['Book-Rating'].value_counts().sort_index()
pd.DataFrame({'Book-Rating': sorted_ratings.index, 'Jumlah': sorted_ratings.values})


# In[17]:


df_ratings = ratings.sample(n=20000, random_state=42)
df_ratings
# df_ratings.to_csv('/content/drive/MyDrive/ML-Project/Deployment/df_ratings.csv', index=False)


# In[18]:


users.info()


# # Data Preprocessing

# In[19]:


books.isnull().sum()


# In[20]:


# menghapus data yang memiliki NaN
clean_books = books.dropna()

clean_books.isnull().sum()


# In[21]:


len(clean_books['ISBN'].unique())


# In[22]:


len(clean_books['Book-Title'].unique())


# In[23]:


users.isnull().sum()


# In[24]:


clean_books = clean_books.drop_duplicates('Book-Title')
clean_books

# clean_books.to_csv('/content/drive/MyDrive/ML-Project/Deployment/clean_books.csv', index=False)


# # **Content Based Filtering**
# 

# In[25]:


books = clean_books[:20000]
books.to_csv('/content/drive/MyDrive/ML-Project/Deployment/books_cbf.csv', index=False)
books = books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author'})

print('Jumlah data buku:', len(books.ISBN.unique()))
print('Jumlah data rating buku dari pembaca:', len(ratings.ISBN.unique()))
print('jumlah data pengguna:', len(users['User-ID'].unique()))

# Books variabel
# Menghapus value pada 'Year-Of-Publication' yang bernilai teks
temp = (books['Year-Of-Publication'] ==
        'DK Publishing Inc') | (books['Year-Of-Publication'] == 'Gallimard')
books = books.drop(books[temp].index)
books[(books['Year-Of-Publication'] == 'DK Publishing Inc')
      | (books['Year-Of-Publication'] == 'Gallimard')]

books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
print(books.dtypes)

books.head()

print("Jumlah nomor ISBN Buku:", len(books['ISBN'].unique()))
print("Jumlah judul buku:", len(books['title'].unique()))
books[(books['Year-Of-Publication'] == 'DK Publishing Inc')]
print('Jumlah penulis buku:', len(books['author'].unique()))
print('Jumlah Tahun Publikasi:', len(books['Year-Of-Publication'].unique()))
print('Jumlah nama penerbit:', len(books['Publisher'].unique()))


# In[26]:


# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()

# Melakukan perhitungan idf pada data book_author
tf.fit(books['author'])

# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names_out()

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(books['author'])

# menyimpan tfidf_matrix
joblib.dump(tfidf_matrix, '/content/drive/MyDrive/ML-Project/Deployment/cbf_tfidf_matrix.pkl')

# menyimpan tfidf_vectorizer
joblib.dump(tf, '/content/drive/MyDrive/ML-Project/Deployment/cbf_tfidf_vectorizer.pkl')

# Melihat ukuran matrix tfidf
tfidf_matrix.shape

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=books.title
).sample(15, axis=1).sample(10, axis=0)


# In[27]:


# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)

# menyimpan index judul buku
joblib.dump(books['title'].tolist(), '/content/drive/MyDrive/ML-Project/Deployment/book_titles.pkl')

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama judul buku
cosine_sim_df = pd.DataFrame(
    cosine_sim, index=books['title'], columns=books['title'])
print('Shape:', cosine_sim_df.shape)

# Melihat similarity matrix pada setiap judul buku
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)


# In[28]:


# Mendapatkan rekomendasi
def book_recommendation(book_title, similarity_data=cosine_sim_df, items=books[['title', 'author']], k=5):
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    if book_title not in similarity_data.columns:
        print(f"Book '{book_title}' not found in the similarity matrix.")
        return pd.DataFrame()

    index = similarity_data.loc[:, book_title].to_numpy(
    ).argpartition(range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop book_title agar nama buku yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(book_title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

# contoh judul buku
book_title_test = "She Said Yes: The Unlikely Martyrdom of Cassie Bernall"

books[books['title'].eq(book_title_test)]

# Mendapatkan rekomendasi judul buku yang mirip
book_recommendation(book_title_test)


# In[29]:


# Evaluasi Model dengan Content Based Filtering
# Menentukan threshold untuk mengkategorikan similarity sebagai 1 atau 0
threshold = 0.5

# Membuat ground truth data dengan asumsi threshold
ground_truth = np.where(cosine_sim >= threshold, 1, 0)

# Menampilkan beberapa nilai pada ground truth matrix
ground_truth_df = pd.DataFrame(ground_truth, index=books['title'], columns=books['title']).sample(5, axis=1).sample(10, axis=0)


# Mengambil sebagian kecil dari cosine similarity matrix dan ground truth matrix
sample_size = 10000
cosine_sim_sample = cosine_sim[:sample_size, :sample_size]
ground_truth_sample = ground_truth[:sample_size, :sample_size]

# Mengonversi cosine similarity matrix menjadi array satu dimensi untuk perbandingan
cosine_sim_flat = cosine_sim_sample.flatten()

# Mengonversi ground truth matrix menjadi array satu dimensi
ground_truth_flat = ground_truth_sample.flatten()

# Menghitung metrik evaluasi
predictions = (cosine_sim_flat >= threshold).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(
    ground_truth_flat, predictions, average='binary', zero_division=1
)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# # NLP dengan BERT/SBERT

# In[30]:


get_ipython().system('pip install -U sentence-transformers')


# Load Data & Model SBERT

# In[31]:


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
books = pd.read_csv('/content/drive/MyDrive/ML-Project/Datasets/Books.csv')

# Cek data
books = books.dropna(subset=['Book-Title'])  # pastikan tidak ada judul kosong

# Load SBERT model (ringan dan cepat)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Embedding Judul Buku

# In[32]:


# Ambil judul buku
book_titles = books['Book-Title'].astype(str).tolist()

# Encode judul buku ke dalam bentuk vektor
title_embeddings = model.encode(book_titles, show_progress_bar=True)


# Fungsi Rekomendasi Berdasarkan Judul

# In[33]:


def recommend_books_by_title(input_title, top_n=5):
    if input_title not in book_titles:
        return f"Buku '{input_title}' tidak ditemukan di dataset."

    input_idx = book_titles.index(input_title)
    input_vec = title_embeddings[input_idx].reshape(1, -1)

    similarities = cosine_similarity(input_vec, title_embeddings)[0]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]

    recommendations = books.iloc[similar_indices][['Book-Title', 'Book-Author', 'Publisher']]
    return recommendations.reset_index(drop=True)


# In[34]:


recommend_books_by_title("Harry Potter and the Chamber of Secrets")


# Evaluasi Model

# Fungsi Evaluasi Lengkap per User

# In[35]:


def evaluate_all_metrics(user_id, ratings_df, books_df, k=5):
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]
    liked_books = user_ratings[user_ratings['Book-Rating'] >= 7]['ISBN'].tolist()

    liked_titles = books_df[books_df['ISBN'].isin(liked_books)]['Book-Title'].values
    if len(liked_titles) == 0:
        return None

    query_title = liked_titles[0]
    recs = recommend_books_by_title(query_title, top_n=k)

    recommended_titles = recs['Book-Title'].tolist()
    relevant_titles = books_df[books_df['ISBN'].isin(liked_books)]['Book-Title'].tolist()

    true_positives = len(set(recommended_titles) & set(relevant_titles))
    relevant_count = len(relevant_titles)

    precision = true_positives / k
    recall = true_positives / relevant_count if relevant_count else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    # MAP@K
    avg_precisions = []
    for i, rec_title in enumerate(recommended_titles):
        if rec_title in relevant_titles:
            avg_precisions.append(1 / (i + 1))
    map_k = sum(avg_precisions) / min(k, relevant_count) if relevant_count else 0

    return {
        "user_id": user_id,
        "query_book": query_title,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "map_at_k": map_k
    }


# Evaluasi Banyak User

# In[36]:


def evaluate_metrics_many_users(user_ids, ratings_df, books_df, k=5):
    results = []
    for user_id in user_ids:
        try:
            result = evaluate_all_metrics(user_id, ratings_df, books_df, k)
            if result is not None:
                results.append(result)
        except:
            continue
    return pd.DataFrame(results)


# In[37]:


user_list = ratings['User-ID'].unique()[:10]
multi_metric_results = evaluate_metrics_many_users(user_list, ratings, books, k=5)
multi_metric_results[['user_id', 'query_book', 'precision_at_k', 'recall_at_k', 'f1_at_k', 'map_at_k']]


# In[38]:


multi_metric_results[['precision_at_k', 'recall_at_k', 'f1_at_k', 'map_at_k']].mean()


# # Data Preparation untuk model Colaborative Filtering

# Encoding

# In[39]:


# fungsi untuk encoding data
def encoding(data_series):
    data = data_series.unique().tolist()
    encoded = {x: i for i, x in enumerate(data)}
    return encoded
# fungsi untuk decoding data
def decoding(data_series):
    data = data_series.unique().tolist()
    decoded = {i: x for i, x in enumerate(data)}
    return decoded

user_encoding = encoding(df_ratings['User-ID'])
isbn_encoding = encoding(df_ratings['ISBN'])

# Menyimpan objek user_encoding ke file .pkl
# joblib.dump(user_encoding, '/content/drive/MyDrive/ML-Project/Deployment/user_encoding.pkl')

# Menyimpan objek user_encoding ke file .pkl
# joblib.dump(isbn_encoding, '/content/drive/MyDrive/ML-Project/Deployment/isbn_encoding.pkl')

df_ratings['user'] = df_ratings['User-ID'].map(user_encoding)
df_ratings['book_title'] = df_ratings['ISBN'].map(isbn_encoding)

df_ratings.head()


# In[40]:


# num_user
num_user = len(user_encoding)
print(f"Number of User : {num_user}")
# num_book_title
num_book = len(isbn_encoding)
print(f"Number of Book : {num_book}")


# In[41]:


# mengubah nilai rating menjadi float
df_ratings['Book-Rating'] = df_ratings['Book-Rating'].values.astype(np.float32)

# nilai minimum rating
min_rating = min(df_ratings['Book-Rating'])
# nilai maksimum rating
max_rating = max(df_ratings['Book-Rating'])

print(df_ratings.shape[0])


# # Collaborative Filtering

# Data Spliting

# In[42]:


x = df_ratings[['user', 'book_title']].values
y = df_ratings['Book-Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


# In[43]:


import tensorflow as tf

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_book, embedding_size, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_book = num_book
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)

        self.book_embedding = tf.keras.layers.Embedding(
            num_book,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.book_bias = tf.keras.layers.Embedding(num_book, 1)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_vector = self.dropout(user_vector)
        user_bias = self.user_bias(inputs[:, 0])

        book_vector = self.book_embedding(inputs[:, 1])
        book_vector = self.dropout(book_vector)
        book_bias = self.book_bias(inputs[:, 1])

        dot_user_book = tf.tensordot(user_vector, book_vector, 2)
        x = dot_user_book + user_bias + book_bias

        return tf.nn.sigmoid(x)


# In[44]:


import tensorflow as tf
# from tensorflow.keras.optimizers import Adam # Remove this import

modelCF = RecommenderNet(num_user, num_book, 50)

# Use the Adam from tensorflow.keras.optimizers
modelCF.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='mse',
                metrics=[tf.keras.metrics.RootMeanSquaredError()],
                run_eagerly=True) # Add this line


# In[45]:


historyCF = modelCF.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=50,
    validation_data=(x_test, y_test)
)


# In[46]:


modelCF.export('/content/drive/MyDrive/ML-Project/Deployment/cf_recommender_model/saved_model')


# Rekomendasi

# In[47]:


book = clean_books.copy()
book = book.rename(columns={'Book-Title': 'title', 'Book-Author': 'author'})

# mengambil sampel user
user_id = df_ratings['User-ID'].sample(1).iloc[0]
book_readed_by_user = df_ratings[df_ratings['User-ID'] == user_id]

# membuat variabel book_not_readed
book_not_readed = book[~book['ISBN'].isin(book_readed_by_user['ISBN'].values)]['ISBN']
book_not_readed = list(
    set(book_not_readed)
    .intersection(set(isbn_encoding.keys()))
)

book_not_readed = [[isbn_encoding.get(x)] for x in book_not_readed]
user_encoder = user_encoding.get(user_id)
user_book_array = np.hstack(
    ([[user_encoder]] * len(book_not_readed), book_not_readed)
)


# In[48]:


ratings_model = modelCF.predict(user_book_array).flatten()

top_ratings_indices = ratings_model.argsort()[-10:][::-1]

recommended_book_ids = [
    book_not_readed[x][0] for x in top_ratings_indices
]

top_book_user = (
    book_readed_by_user.sort_values(
        by='Book-Rating',
        ascending=False
    )
    .head(10)['ISBN'].values
)

book_rows = book[book['ISBN'].isin(top_book_user)]

# Menampilkan rekomendasi buku dalam bentuk DataFrame
book_rows_data = []
for row in book_rows.itertuples():
    book_rows_data.append([row.title, row.author])

recommended_isbn = [list(isbn_encoding.keys())[i] for i in recommended_book_ids]
recommended_book = book[book['ISBN'].isin(recommended_isbn)]

recommended_book_data = []
for row in recommended_book.itertuples():
    recommended_book_data.append([row.title, row.author])

# Membuat DataFrame untuk output
output_columns = ['Book Title', 'Book Author']
df_book_readed_by_user = pd.DataFrame(book_rows_data, columns=output_columns)
df_recommended_books = pd.DataFrame(recommended_book_data, columns=output_columns)

# Menampilkan hasil rekomendasi dalam bentuk DataFrame
print("Showing recommendation for users: {}".format(user_id))
print("===" * 9)
print("Book with high ratings from user")
print("----" * 8)
print(df_book_readed_by_user)
print("----" * 8)
print("Top 10 books recommendation")
print("----" * 8)
df_recommended_books


# In[49]:


plt.plot(historyCF.history['root_mean_squared_error'])
plt.plot(historyCF.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
