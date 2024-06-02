import pandas as pd
import csv
import snscrape.modules.telegram as sntg

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Замените 'channel_name' на имя вашего канала
channel_name = 'DtRoad'

# Создаем экземпляр класса TelegramChannelScraper
scraper = sntg.TelegramChannelScraper(channel_name)

# Открываем CSV-файл для записи
with open('postse.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # Записываем заголовок
    writer.writerow(['id', 'date', 'post_text'])

    # Извлекаем данные из канала
    for post in scraper.get_items():
        # Извлекаем id, дату и текст поста
        post_id = post.url
        post_date = post.date
        post_text = post.content

        # Записываем данные в CSV-файл
        writer.writerow([post_id, post_date, post_text])


data = pd.read_csv('postsе.csv')
print(data)