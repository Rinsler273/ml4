import csv


def get_data_types(csv_file):
    # Создаем словарь для хранения типов данных
    data_types = {}

    # Открываем CSV файл для чтения с явным указанием кодировки
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)

        # Читаем заголовки столбцов
        headers = next(reader)

        # Инициализируем типы данных для каждого столбца
        for header in headers:
            data_types[header] = set()

        # Читаем данные из каждой строки файла
        for row in reader:
            for i, value in enumerate(row):
                # Определяем тип данных и добавляем его в словарь
                if value.isdigit():
                    data_types[headers[i]].add(int)
                elif value.replace('.', '', 1).isdigit():
                    data_types[headers[i]].add(float)
                else:
                    data_types[headers[i]].add(str)

    return data_types


# Имя CSV файла с данными
input_csv = 'postse.csv'

# Получаем типы данных из CSV файла
types = get_data_types(input_csv)

# Выводим типы данных на экран
for column, data_type in types.items():
    print("Тип данных для столбца '{}':".format(column), end=' ')
    if int in data_type:
        print("int", end='')
    if float in data_type:
        print(", float", end='')
    if str in data_type:
        print(", str", end='')
    print()

