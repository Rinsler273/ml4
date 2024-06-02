import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords

# Загрузка данных из CSV
df = pd.read_csv('posts2.csv')

# Заполнение пропущенных значений и приведение к строковому типу
df['post_text'] = df['post_text'].fillna('').astype(str)

# Предобработка текста
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

def clean_text(text):
    # Удаление смайликов
    text = re.sub(r'[^\w\s,]', '', text)
    # Удаление союзов и стоп-слов
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['post_text'] = df['post_text'].apply(clean_text)

# Подготовка данных
texts = df['post_text'].values
labels = df['0'].values

# Преобразование меток в категориальный формат
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

# Токенизация текста
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Паддинг последовательностей
max_sequence_length = 20
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Создание модели
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Вычисление матрицы ошибок
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Отображение матрицы ошибок
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(df['0']), yticklabels=np.unique(df['0']))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Отчет классификации
report = classification_report(y_true, y_pred_classes, target_names=[f'Chat {i}' for i in np.unique(df['0'])], output_dict=True)
print(classification_report(y_true, y_pred_classes, target_names=[f'Chat {i}' for i in np.unique(df['0'])]))

# Тестирование модели на новых данных
def predict_chat(model, tokenizer, text, max_sequence_length):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    pred = model.predict(padded_sequence)
    return np.argmax(pred, axis=1)[0]

# Пример использования для предсказания нового сообщения
new_message = " «Победа» ввела новые надбавки для бортпроводников. С июня авиакомпания начнет доплачивать бортпроводникам за стаж работы. У тех, кто проработал больше года, заработная плата увеличится на 7 650 рублей. Еще одну надбавку в размере 4 тыс. рублей лоукостер ввел для членов кабинных экипажей, летающих из Шереметьево. В качестве компенсации оплаты Аэроэкспресса до аэропорта. Кроме того, позитивные изменения коснулись компенсации бортпроводникам питания. «Победа» увеличила ее почти на 13% — с 140 до 158 рублей за каждый полетный час."
predicted_chat = predict_chat(model, tokenizer, new_message, max_sequence_length)
print(f'The predicted chat for the message "{new_message}" is Chat {predicted_chat}')