import csv

def count_values_in_csv(file_path, delimiter=','):
    # Считываем данные из CSV файла с указанием кодировки UTF-8
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        data = list(reader)

    # Подсчитываем количество значений в считанных данных
    num_values = sum(len(row) for row in data)

    return num_values

# Пример использования
num_values = count_values_in_csv('posts1.csv', ';')
print(f'Количество значений в файле : {num_values}')


