import struct
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#test
def parse_bin_file(file_path): #Функция для парса bin файла
    # Формат: prefix(4B), deviceId(I), channelId(H), timestamp(Q),
    # cfd_y1(h), cfd_y2(h), heigth(h), baseline(h),
    # qLong(i), qShort(i), psdValue(h), eventCounter(I),
    # eventCounterPSD(I), decimationFactor(H), postfix(4B)
    record_format = '<4B I H Q h h h h i i h I I H 4B'  # 56 байт
    record_size = struct.calcsize(record_format)

    parsed_records = []
    with open(file_path, 'rb') as f:
        f.read(8)  # пропускаем заголовок (8 байт)

        while True:
            chunk = f.read(record_size)
            if len(chunk) < record_size:
                break
            values = struct.unpack(record_format, chunk)
            # # Разбиваем сырые байты по полям
            # splitted = split_raw_bytes_by_format(chunk, record_format[1:])  # убираем "<"
            # print("Raw bytes split by fields:")
            # for b, fmt_code, size in splitted:
            #     print(f"{fmt_code:5}: {b.hex()}  ({size} bytes)")
            # print("Parsed values:", values)
            # print("-" * 40)
            parsed_records.append(values)

    return parsed_records

def save_to_csv(records, bin_filename): # Функция для сохранения результатов парса в csv
    csv_filename = Path(bin_filename).with_suffix('.csv')

    header = [
        'title1_4', 'title2_4', 'title3_4', 'title4_4',
        'deviceId', 'channelId', 'timestamp',
        'cfd_y1', 'cfd_y2', 'heigth', 'baseline',
        'qLong', 'qShort', 'psdValue',
        'eventCounter', 'eventCounterPSD', 'decimationFactor',
        'postfix1', 'postfix2', 'postfix3', 'postfix4'
    ]

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)

# def split_raw_bytes_by_format(raw_bytes, fmt):
#     """
#     Разбивает raw_bytes согласно fmt на куски, соответствующие каждому полю.
#     Возвращает список кортежей (field_bytes, field_name, size).
#     """
#     sizes = []
#     i = 0
#     # Разбор формата на части с учётом повторений, например '4B', 'I', 'H' и т.п.
#     import re
#     pattern = re.compile(r'(\d*)([xcbBhHiIlLqQnPfd])')
#
#     fields = []
#     for count, code in pattern.findall(fmt):
#         count = int(count) if count else 1
#         size = struct.calcsize(f'{count}{code}')
#         fields.append((count, code, size))
#
#     result = []
#     pos = 0
#     for count, code, size in fields:
#         field_bytes = raw_bytes[pos:pos + size]
#         result.append((field_bytes, f'{count if count > 1 else ""}{code}', size))
#         pos += size
#     return result

def plot_specta_from_csv(csv_file): # Функция для построения спектра отклика
    qlong_values = []

    # Считываем qLong
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qlong_values.append(int(row['qLong']))

    qlong_values = np.array(qlong_values)
    q_min = qlong_values.min()
    q_max = qlong_values.max()

    # Нормируем значения qLong в диапазон [0, 4095], затем округляем до целых бинов [0, 4095]
    normalized_bins = np.floor((qlong_values - q_min) / (q_max - q_min) * 4095).astype(int)

    # Строим гистограмму: 4096 бинов, индексы от 0 до 4095 → X от 1 до 4096
    counts = np.bincount(normalized_bins, minlength=4096)
    bin_numbers = np.arange(1, 4097)  # от 1 до 4096 включительно

    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.bar(bin_numbers, counts, width=1.0, edgecolor='black')
    plt.xlabel('Номер канала')
    plt.ylabel('Количество событий')
    plt.title('Спектр отклика')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    # Кастомный шаг по х
    plt.xticks(
        ticks=np.arange(0, 4096, 250),
        labels=np.arange(0, 4096, 250)
    )
    plt.tight_layout()

def plot_psd_from_csv(csv_file, num_bins=200): #Функция для построения PSD-диаграммы
    qlong = []
    qshort = []

    # Читаем значения из CSV
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ql = int(row['qLong'])
            qs = int(row['qShort'])

            # Пропускаем некорректные значения, чтобы избежать деления на 0
            if ql > 0:
                qlong.append(ql)
                qshort.append(qs)

    qlong = np.array(qlong)
    qshort = np.array(qshort)

    # Вычисляем PSD = 1 - qShort / qLong
    psd = 1 - (qshort / qlong)

    # Фильтрация NaN, inf, -inf (если все же остались)
    psd = psd[np.isfinite(psd)]

    # Строим гистограмму
    plt.figure(figsize=(10, 6))
    plt.hist(psd, bins=num_bins, range=(0,1), edgecolor='black')
    plt.xlabel('PSD')
    plt.ylabel('Количество событий')
    plt.title('PSD-диаграмма')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

# Основной код
bin_file = '/Users/babich/PycharmProjects/api/data_psd_2025_07_07__16_20_00.bin'
records = parse_bin_file(bin_file)
save_to_csv(records, bin_file)
csv_filename = Path(bin_file).with_suffix('.csv')
plot_specta_from_csv(csv_filename)
plot_psd_from_csv(csv_filename)
plt.show()