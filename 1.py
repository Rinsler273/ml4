import csv

def change_values_in_csv_column(file_path, column_index, new_value):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Меняем значения в определенном столбце считанных данных
    for row in data:
        row[column_index] = new_value

    # Записываем измененные данные в CSV файл с указанием кодировки UTF-8
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def merge_csv_files(file_path_1, file_path_2, output_file_path):
    # Считываем данные из первого CSV файла с указанием кодировки UTF-8
    with open(file_path_1, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data_1 = list(reader)

    # Считываем данные из второго CSV файла с указанием кодировки UTF-8
    with open(file_path_2, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data_2 = list(reader)

    # Объединяем данные из двух CSV файлов
    data = data_1 + data_2

    # Записываем объединенные данные в новый CSV файл с указанием кодировки UTF-8
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def remove_last_n_rows(file_path, n):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Удаляем последние n строк из данных
    if len(data) > n:
        data = data[:-n]
    else:
        data = []

    # Записываем измененные данные обратно в CSV файл с указанием кодировки UTF-8
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Пример использования
change_values_in_csv_column('posts.csv', 0, '1')
change_values_in_csv_column('posts1.csv', 0, '0')
remove_last_n_rows('posts.csv', 4)
remove_last_n_rows('posts1.csv', 4)
merge_csv_files('posts1.csv', 'posts.csv', 'posts2.csv')

