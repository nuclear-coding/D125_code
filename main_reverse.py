import struct
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# === PSD THRESHOLD SETTING ===
PSD_THRESHOLD = 0.1474  # Change this value to update threshold everywhere


# === PSD THRESHOLD SETTING ===


def parse_bin_file(file_path):
    """
    Parse a binary file into a list of records.
    Format: prefix(4B), deviceId(I), channelId(H), timestamp(Q),
    cfd_y1(h), cfd_y2(h), heigth(h), baseline(h),
    qLong(i), qShort(i), psdValue(h), eventCounter(I),
    eventCounterPSD(I), decimationFactor(H), postfix(4B)
    """
    record_format = '<4B I H Q h h h h i i h I I H 4B'  # 56 bytes
    record_size = struct.calcsize(record_format)
    parsed_records = []

    with open(file_path, 'rb') as f:
        f.read(8)  # skip header (8 bytes)
        while True:
            chunk = f.read(record_size)
            if len(chunk) < record_size:
                break
            values = struct.unpack(record_format, chunk)
            parsed_records.append(values)
    return parsed_records


def save_to_csv(records, bin_filename):
    """
    Save parsed records to a CSV file.
    Each value is written to its own cell (column), not as a string with separators.
    """
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
        writer = csv.writer(f, delimiter=',')  # Ensure comma delimiter
        writer.writerow(header)
        for row in records:
            # If any value is a tuple or list, flatten it; otherwise, just write the row
            if isinstance(row, (tuple, list)):
                writer.writerow(list(row))
            else:
                writer.writerow([row])


def plot_specta_from_csv(csv_file, filter_psd_threshold=True):
    """
    Plot the response spectrum from CSV data, with optional PSD filtering.
    """
    qlong_values, qshort_values = [], []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qlong_values.append(int(row['qLong']))
            qshort_values.append(int(row['qShort']))
    qlong_values = np.array(qlong_values)
    qshort_values = np.array(qshort_values)
    mask = (qlong_values > 0) & (qshort_values >= 0)
    qlong_values = qlong_values[mask]
    qshort_values = qshort_values[mask]
    print(f"Всего точек в спектре: {len(qlong_values)}")

    if filter_psd_threshold:
        valid = qlong_values > 0
        qlong_valid = qlong_values[valid]
        qshort_valid = qshort_values[valid]
        psd = 1 - (qshort_valid / qlong_valid)
        mask = (psd > PSD_THRESHOLD) & np.isfinite(psd)
        qlong_valid = qlong_valid[mask]
        qshort_valid = qshort_valid[mask]
        print(f"После фильтрации PSD > {PSD_THRESHOLD}: {len(qlong_valid)}")
        qlong_values = qlong_valid
        qshort_values = qshort_valid
    else:
        qlong_values = qlong_values[qlong_values > 0]
    if len(qlong_values) == 0:
        print("Нет данных для построения спектра.")
        return
    q_min, q_max = qlong_values.min(), qlong_values.max()
    normalized_bins = np.floor((qlong_values - q_min) / (q_max - q_min) * 4095).astype(int)
    counts = np.bincount(normalized_bins, minlength=4096)
    bin_numbers = np.arange(1, 4097)
    plt.figure(figsize=(12, 6))
    plt.bar(bin_numbers, counts, width=1.0, edgecolor='black')
    plt.xlabel('Номер канала')
    plt.ylabel('Количество событий')
    plt.yscale('log')
    plt.title(f"Спектр отклика (PSD фильтр = {filter_psd_threshold})")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.xticks(
        ticks=np.arange(0, 4096, 250),
        labels=[str(x) for x in np.arange(0, 4096, 250)]
    )
    plt.tight_layout()


def plot_psd_from_csv(csv_file, num_bins=200, filter_psd_threshold=True):
    """
    Plot the PSD diagram from CSV data, with optional PSD filtering.
    """
    qlong_values, qshort_values = [], []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ql = int(row['qLong'])
            qs = int(row['qShort'])
            if ql > 0:
                qlong_values.append(ql)
                qshort_values.append(qs)
    qlong_values = np.array(qlong_values)
    qshort_values = np.array(qshort_values)
    mask = (qlong_values > 0) & (qshort_values >= 0)
    qlong_values = qlong_values[mask]
    qshort_values = qshort_values[mask]
    psd = 1 - (qshort_values / qlong_values)

    if filter_psd_threshold:
        mask = psd > PSD_THRESHOLD
        psd = psd[mask]
        print(f"После фильтрации PSD > {PSD_THRESHOLD}: {len(psd)}")
    psd = psd[np.isfinite(psd)]
    plt.figure(figsize=(10, 6))
    plt.hist(psd, bins=num_bins, range=(0, 1), edgecolor='black')
    plt.xlabel('PSD')
    plt.ylabel('Количество событий')
    plt.title(f"PSD-диаграмма (PSD фильтр = {filter_psd_threshold})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()


def save_spectra_to_csv(bin_centers, counts_unfiltered, counts_filtered, output_file, bin_filename=None):
    """
    Save spectra bin centers and counts (unfiltered and filtered) to a CSV file.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_center', 'count_unfiltered', 'count_filtered'])
        for b, cu, cf in zip(bin_centers, counts_unfiltered, counts_filtered):
            writer.writerow([b, cu, cf])


def save_psd_to_csv(psd_unfiltered, psd_filtered, output_file):
    """
    Save PSD values (unfiltered and filtered) to a CSV file.
    Each row contains one value from each array, in separate columns.
    """
    max_len = max(len(psd_unfiltered), len(psd_filtered))
    # Pad shorter array with empty values (NaN)
    psd_unfiltered = np.pad(psd_unfiltered, (0, max_len - len(psd_unfiltered)), constant_values=np.nan)
    psd_filtered = np.pad(psd_filtered, (0, max_len - len(psd_filtered)), constant_values=np.nan)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')  # Ensure comma delimiter
        writer.writerow(['psd_unfiltered', 'psd_filtered'])
        for pu, pf in zip(psd_unfiltered, psd_filtered):
            writer.writerow([pu, pf])


def save_psd_hist_to_csv(bin_centers, counts_unfiltered, counts_filtered, output_file, bin_filename=None):
    """
    Save PSD histogram bin centers and counts (unfiltered and filtered) to a CSV file.
    Each row contains one bin center and the counts for both cases.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['bin_center', 'count_unfiltered', 'count_filtered'])
        for b, cu, cf in zip(bin_centers, counts_unfiltered, counts_filtered):
            writer.writerow([b, cu, cf])


def plot_spectra_comparison(csv_file):
    """
    Plot both filtered and unfiltered response spectra on the same figure for comparison.
    Uses line plots for clear color/legend correspondence.
    Also saves the plotted data to a CSV file.
    Prints the number of points for each case.
    The x-axis (qlong) always starts at 0 and ends at 200,000, with 4096 bins.
    """
    qlong_values, qshort_values = [], []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qlong_values.append(int(row['qLong']))
            qshort_values.append(int(row['qShort']))
    qlong_values = np.array(qlong_values)
    qshort_values = np.array(qshort_values)
    mask = (qlong_values > 0) & (qshort_values >= 0)
    qlong_values = qlong_values[mask]
    qshort_values = qshort_values[mask]

    # Тут изменение шкалы Х: 4096 бинов от qlong_min до qlong_max
    qlong_min = 0
    qlong_max = 100_000
    num_bins = 4096
    bin_edges = np.linspace(qlong_min, qlong_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Unfiltered
    qlong_unfiltered = qlong_values[qlong_values > 0]
    print(f"Спектр: всего точек без фильтра: {len(qlong_unfiltered)}")
    counts_unfiltered, _ = np.histogram(qlong_unfiltered, bins=bin_edges)

    # Filtered
    valid = qlong_values > 0
    qlong_valid = qlong_values[valid]
    qshort_valid = qshort_values[valid]
    psd = 1 - (qshort_valid / qlong_valid)
    mask_psd = (psd > PSD_THRESHOLD) & np.isfinite(psd)
    qlong_filtered = qlong_valid[mask_psd]
    print(f"Спектр: всего точек с фильтром PSD > {PSD_THRESHOLD}: {len(qlong_filtered)}")
    counts_filtered, _ = np.histogram(qlong_filtered, bins=bin_edges)

    # Save spectra data
    bin_file_name = Path(csv_file).with_suffix('.bin').name
    spectra_csv = Path(csv_file).with_name(f'spectra_plot_data_{bin_file_name}.csv')
    save_spectra_to_csv(bin_centers, counts_unfiltered, counts_filtered, spectra_csv, bin_filename=bin_file_name)

    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers, counts_unfiltered, color='blue', label='Без фильтра PSD')
    plt.plot(bin_centers, counts_filtered, color='red', label=f'С фильтром PSD > {PSD_THRESHOLD}')
    plt.xlabel('qLong')
    plt.ylabel('Количество событий')
    plt.yscale('log')
    plt.xlim(qlong_min, qlong_max)
    plt.title('Сравнение спектров отклика (без фильтра и с фильтром PSD)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)

    # ТУТ МЕНЯТЬ ШАГ ПО ОСИ X
    plt.xticks(
        ticks=np.arange(0, qlong_max + 1, 5_000),
        labels=[str(x) for x in np.arange(0, qlong_max + 1, 5_000)]
    )
    # ТУТ МЕНЯТЬ ШАГ ПО ОСИ Х

    plt.legend()
    plt.tight_layout()


def plot_psd_comparison(csv_file, num_bins=200):
    """
    Plot both filtered and unfiltered PSD histograms on the same figure for comparison.
    Colors in the legend match the histogram colors.
    Also saves the plotted histogram data to a CSV file (bin centers and counts).
    Prints the number of points for each case.
    """
    qlong_values, qshort_values = [], []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ql = int(row['qLong'])
            qs = int(row['qShort'])
            if ql > 0:
                qlong_values.append(ql)
                qshort_values.append(qs)
    qlong_values = np.array(qlong_values)
    qshort_values = np.array(qshort_values)
    mask = (qlong_values > 0) & (qshort_values >= 0)
    qlong_values = qlong_values[mask]
    qshort_values = qshort_values[mask]
    psd = 1 - (qshort_values / qlong_values)
    psd = psd[np.isfinite(psd)]

    # Unfiltered
    psd_unfiltered = psd
    print(f"PSD: всего точек без фильтра: {len(psd_unfiltered)}")
    # Filtered
    psd_filtered = psd[psd > PSD_THRESHOLD]
    print(f"PSD: всего точек с фильтром PSD > {PSD_THRESHOLD}: {len(psd_filtered)}")

    # Compute histograms for both cases
    counts_unfiltered, bin_edges = np.histogram(psd_unfiltered, bins=num_bins, range=(0, 1))
    counts_filtered, _ = np.histogram(psd_filtered, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Save PSD histogram data
    bin_file_name = Path(csv_file).with_suffix('.bin').name
    psd_csv = Path(csv_file).with_name(f'psd_plot_data_{bin_file_name}.csv')
    save_psd_hist_to_csv(bin_centers, counts_unfiltered, counts_filtered, psd_csv, bin_filename=bin_file_name)

    plt.figure(figsize=(10, 6))
    plt.hist(psd_unfiltered, bins=bin_edges.tolist(), edgecolor='black', alpha=0.5, color='blue',
             label='Без фильтра PSD')
    plt.hist(psd_filtered, bins=bin_edges.tolist(), edgecolor='red', alpha=0.5, color='red',
             label=f'С фильтром PSD > {PSD_THRESHOLD}')
    plt.xlabel('PSD')
    plt.ylabel('Количество событий')
    plt.title('Сравнение PSD-диаграмм (без фильтра и с фильтром PSD)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()


def main():
    bin_file = '/Users/babich/PycharmProjects/api/data_psd_ING07T-300s_117kV_2025_07_08__17_04_54.bin'
    records = parse_bin_file(bin_file)
    save_to_csv(records, bin_file)
    csv_filename = Path(bin_file).with_suffix('.csv')
    # Plot spectra comparison
    plot_spectra_comparison(csv_filename)
    # Plot PSD comparison
    plot_psd_comparison(csv_filename)
    plt.show()


if __name__ == "__main__":
    main()