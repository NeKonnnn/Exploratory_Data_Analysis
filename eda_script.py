import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from scipy.stats import kurtosis, skew, norm
from scipy.stats import gaussian_kde
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from phik.report import plot_correlation_matrix
from phik import phik_matrix
from phik.binning import bin_data

import time

def timing(func):
    """
    Декоратор для измерения времени выполнения функций.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Функция '{func.__name__}' выполнена за {elapsed_time:.2f} секунд.")
        return result
    return wrapper

class EDAProcessor:
    def __init__(self, df, output_dir="DATA_OUT/graphics"):
        """
        Инициализация класса для выполнения разведочного анализа данных.

        Параметры:
        df (pd.DataFrame): Входной DataFrame для анализа.
        output_dir (str): Директория для сохранения графиков.
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("DATA_OUT", exist_ok=True)

    @timing
    def generate_eda_summary(self):
        """
        Сформировать сводную таблицу для разведочного анализа данных.

        Возвращает:
        dict: Словарь с четырьмя ключами:
            - 'summary': DataFrame с названиями столбцов, количеством пропусков, процентом пропусков, типами данных, количеством уникальных значений и рекомендациями.
            - 'duplicate_count': Общее количество дубликатов в DataFrame.
            - 'duplicates': DataFrame с дублированными строками (если они есть).
            - 'duplicate_columns': Список дублированных столбцов (если такие есть).
        """
        summary = pd.DataFrame({
            'Название столбца': self.df.columns,
            'Пропущено строк': self.df.isnull().sum(),
            'Процент пропусков, %': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'Тип данных': self.df.dtypes,
            'Количество уникальных значений': [self.df[col].nunique() for col in self.df.columns],
            'Уникальные (категориальные) значения': [
                self.df[col].dropna().unique().tolist() if self.df[col].dtype in ['object', 'category', 'string'] else None
                for col in self.df.columns
            ],
            'Рекомендации': [
                "Удалить (константный столбец)" if self.df[col].nunique() == 1 else
                "Удалить (ID или уникальные значения)" if self.df[col].nunique() == len(self.df) else
                "Оставить"
                for col in self.df.columns
            ]
        }).reset_index(drop=True)

        duplicate_count = self.df.duplicated().sum()
        duplicates = self.df[self.df.duplicated()] if duplicate_count > 0 else pd.DataFrame()

        # Поиск дублированных столбцов
        duplicate_columns = []
        for i, col1 in enumerate(self.df.columns):
            for col2 in self.df.columns[i + 1:]:
                if self.df[col1].equals(self.df[col2]):
                    duplicate_columns.append((col1, col2))

        return {
            'summary': summary,
            'duplicate_count': duplicate_count,
            'duplicates': duplicates,
            'duplicate_columns': duplicate_columns
        }

    @timing
    def plot_target_distribution(self, target_column):
        """
        Построить график распределения бинарной целевой переменной и сохранить его в файл.

        Параметры:
        target_column (str): Название колонки с целевой переменной.
        """
        target_counts = self.df[target_column].value_counts()
        ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette="viridis")
        plt.title("Распределение целевой переменной")
        plt.xlabel("Значение целевой переменной")
        plt.ylabel("Количество")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
        file_path = os.path.join(self.output_dir, f"target_distribution_{target_column}.jpg")
        plt.savefig(file_path, bbox_inches='tight')
        plt.show()
        plt.close()

        # Создание сводной таблицы
        target_summary = pd.DataFrame({
            'Значение': target_counts.index,
            'Количество': target_counts.values
        })
        return target_summary

    @timing
    def plot_categorical_distributions(self, categorical_columns):
        """
        Построить графики распределения для всех категориальных переменных и сохранить их в файлы.

        Параметры:
        categorical_columns (list): Список категориальных колонок.
        """
        category_summary = []

        for column in categorical_columns:
            # Удаляем пропуски
            non_na_data = self.df[column].dropna()

            # Если после удаления пропусков данных нет, пропускаем
            if non_na_data.empty:
                print(f"Пропущен график для столбца {column}, так как он пустой после удаления пропусков.")
                continue

            # Проверяем формат данных: даты
            try:
                parsed_dates = pd.to_datetime(non_na_data, format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:  # Если более 80% значений интерпретируются как даты
                    print(f"Пропущен график для столбца {column}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {column}: {e}")
                continue

            # Проверяем тип данных: исключаем числовые столбцы
            if pd.api.types.is_numeric_dtype(non_na_data):
                print(f"Пропущен график для столбца {column}, так как он содержит числовые данные.")
                continue

            # Количество уникальных значений
            unique_count = non_na_data.nunique()

            # Автоматический подбор размеров графика
            width = min(max(8, unique_count * 0.5), 20)  # Масштабируем ширину графика
            rotation = 0 if unique_count <= 5 else 30 if unique_count <= 10 else 60  # Автоматический угол поворота подписей
            height = 6  # Фиксированная высота

            # Строим график
            plt.figure(figsize=(width, height))
            ax = sns.countplot(x=non_na_data.astype(str), palette="viridis")
            plt.title(f"Распределение категориальной переменной: {column}")
            plt.xlabel(column)
            plt.ylabel("Количество")
            plt.xticks(rotation=rotation)

            # Аннотации
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

            # Сохранение графика
            file_path = os.path.join(self.output_dir, f"categorical_distribution_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

            # Добавляем данные в сводную таблицу
            column_summary = non_na_data.value_counts().reset_index()
            column_summary.columns = ['Элемент', 'Количество']
            column_summary.insert(0, 'Признак', column)
            category_summary.append(column_summary)

        # Объединяем все данные в одну таблицу
        if category_summary:
            formatted_summary = pd.concat(category_summary, ignore_index=True)
            formatted_summary['Признак'] = formatted_summary['Признак'].where(~formatted_summary['Признак'].duplicated(keep='first'), '')
            return formatted_summary
        else:
            print("Нет подходящих категориальных переменных для анализа.")
            return pd.DataFrame()

    @timing
    def plot_categorical_vs_target(self, target_column, categorical_columns):
        """
        Построить графики сравнения категориальных переменных и целевой переменной,
        сохранить их в файлы и вернуть сводную таблицу распределения.

        Параметры:
        target_column (str): Название колонки с целевой переменной.
        categorical_columns (list): Список категориальных колонок.

        Возвращает:
        pd.DataFrame: Сводная таблица распределения категориальных признаков по значениям целевой переменной.
        """
        summary_data = []

        for column in categorical_columns:
            # Удаляем пропуски в текущем столбце и целевой переменной
            non_na_data = self.df[[column, target_column]].dropna()

            if non_na_data.empty:
                print(f"Пропущен график для столбца {column}, так как он пустой после удаления пропусков.")
                continue

            # Проверяем формат данных: даты
            try:
                parsed_dates = pd.to_datetime(non_na_data[column], format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:
                    print(f"Пропущен график для столбца {column}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {column}: {e}")
                continue

            # Проверяем тип данных: исключаем числовые столбцы
            if pd.api.types.is_numeric_dtype(non_na_data[column]):
                print(f"Пропущен график для столбца {column}, так как он содержит числовые данные.")
                continue

            # Количество уникальных значений
            unique_count = non_na_data[column].nunique()
            
            # Автоматический подбор размеров графика
            width = min(max(8, unique_count * 0.5), 20)  # Масштабируем ширину графика
            rotation = 0 if unique_count <= 5 else 30 if unique_count <= 10 else 60  # Автоматический угол поворота подписей
            height = 6  # Фиксированная высота

            # Строим график
            plt.figure(figsize=(width, height))
            ax = sns.countplot(
                x=non_na_data[column].astype(str),
                hue=non_na_data[target_column].astype(str),
                palette="viridis"
            )
            plt.title(f"Распределение {column} в зависимости от {target_column}")
            plt.xlabel(column)
            plt.ylabel("Количество")
            plt.xticks(rotation=rotation)
            plt.legend(title=target_column)

            # Аннотации
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

            # Сохранение графика
            file_path = os.path.join(self.output_dir, f"categorical_vs_target_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

            # Создаем сводную таблицу распределения
            category_distribution = non_na_data.groupby([column, target_column]).size().reset_index(name='Количество')
            category_distribution.insert(0, 'Признак', column)
            summary_data.append(category_distribution)

        # Объединяем все данные в один DataFrame
        if summary_data:
            summary_df = pd.concat(summary_data, ignore_index=True)
            return summary_df
        else:
            print("Нет подходящих категориальных переменных для анализа.")
            return pd.DataFrame()

    @timing
    def plot_kde_distributions(self, numeric_columns):
        """
        Построить гистограммы и графики KDE для всех числовых переменных и аналитикой,
        включая идеальное нормальное распределение.

        Параметры:
        numeric_columns (list): Список числовых колонок.

        Возвращает:
        pd.DataFrame: Итоговая сводная таблица с аналитикой по признакам.
        """

        summary_data = []

        for column in numeric_columns:
            data = self.df[column].dropna()

            # Проверяем уникальность значений
            if data.nunique() <= 1:  # Если в данных только одно уникальное значение
                print(f"Пропущен график для столбца {column}, так как все значения одинаковы или отсутствует дисперсия.")
                continue

            plt.figure(figsize=(10, 6))
            # Построение гистограммы с KDE
            ax = sns.histplot(
                data,
                kde=True,
                bins=30,
                edgecolor="black",
                color="royalblue",
                alpha=0.8,
                linewidth=1.2
            )

            # Изменение цвета KDE линии
            kde_color = "darkorange"
            sns.kdeplot(data, color=kde_color, linewidth=2, label="KDE (плотность)")

            plt.title(f"Гистограмма и KDE для признака: {column}", fontsize=14, fontweight="bold")
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Плотность / Частота", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Аннотация медианы и среднего
            median = data.median()
            mean = data.mean()
            std_dev = data.std()
            plt.axvline(median, color="red", linestyle="--", linewidth=2, label=f"Медиана: {median:.2f}")
            plt.axvline(mean, color="blue", linestyle="-.", linewidth=2, label=f"Среднее: {mean:.2f}")

            # Добавляем перцентильные линии
            percentiles = [0.25, 0.75, 0.99]
            percentile_values = data.quantile(percentiles)
            perc_colors = ["green", "purple", "brown"]  # Цвета для перцентилей
            for perc, value, color in zip(percentiles, percentile_values, perc_colors):
                plt.axvline(value, color=color, linestyle=":", linewidth=2, label=f"{int(perc * 100)}-й перцентиль: {value:.2f}")

            # Добавляем идеальное нормальное распределение
            try:
                x_range = np.linspace(data.min(), data.max(), 1000)  # Диапазон значений
                ideal_pdf = norm.pdf(x_range, loc=mean, scale=std_dev)  # Плотность вероятности нормального распределения
                ideal_pdf_scaled = ideal_pdf * len(data) * (data.max() - data.min()) / 30  # Масштабируем для совпадения с гистограммой
                plt.plot(x_range, ideal_pdf_scaled, color="orange", linestyle="--", linewidth=2.5, label="Идеальное распределение")
            except Exception as e:
                print(f"Ошибка при построении идеального распределения для столбца {column}: {e}")

            # Вычисляем пиковое значение KDE
            try:
                kde = gaussian_kde(data)
                kde_values = kde(x_range)
                peak_x = x_range[np.argmax(kde_values)]
                peak_y = kde_values.max()

                # Аннотация пика
                plt.annotate(f"Пик: {peak_x:.2f}", xy=(peak_x, peak_y),
                            xytext=(peak_x + 0.5, peak_y + 0.1),
                            arrowprops=dict(facecolor='black', arrowstyle="->"),
                            fontsize=10)
            except np.linalg.LinAlgError:
                print(f"Не удалось построить KDE для столбца {column}, так как данные имеют низкую дисперсию.")
                peak_x = None

            # Легенда
            plt.legend(fontsize=10)

            # Сохранение графика
            file_path = os.path.join(self.output_dir, f"kde_distribution_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

            # Расчет статистик
            kurt = kurtosis(data)
            skewness = skew(data)

            # Определение распределения
            if -0.5 <= skewness <= 0.5:
                distribution = "Нормальное"
            elif skewness > 0.5:
                distribution = "Смещенное вправо"
            elif skewness < -0.5:
                distribution = "Смещенное влево"
            else:
                distribution = "Неопределено"

            # Интервал, в котором распределено большинство значений (межквартильный диапазон)
            lower, upper = data.quantile(0.25), data.quantile(0.75)
            range_info = f"[{lower:.2f}, {upper:.2f}]"

            # Сохранение данных в итоговую таблицу
            summary_data.append({
                "Признак": column,
                "Эксцесс": round(kurt, 2),
                "Асимметрия": round(skewness, 2),
                "Пик": round(peak_x, 2) if peak_x else None,
                "Среднее": round(mean, 2),
                "Медиана": round(median, 2),
                "25-й перцентиль": round(percentile_values[0.25], 2),
                "75-й перцентиль": round(percentile_values[0.75], 2),
                "99-й перцентиль": round(percentile_values[0.99], 2),
                "Распределение": range_info,
                "Вывод": distribution
            })

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    @timing
    def detect_outliers_iqr(self, numeric_columns):
        """
        Функция для построения графика "Ящик с усами" для числовых переменных и сохранения их в файлы, а также определения выбросов.

        Параметры:
        numeric_columns (list): Список числовых колонок.

        Возвращает:
        pd.DataFrame: Сводная таблица с признаками и выбросами.
        """
        outlier_summary = []

        for column in numeric_columns:
            # Вычисляем квартильные значения
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Определяем границы выбросов
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Находим выбросы
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)][column].tolist()

            # Добавляем данные о выбросах в сводную таблицу
            outlier_summary.append({
                'Признак': column,
                'Выбросы': outliers
            })

            # Построение графика
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[column], palette="viridis")
            plt.title(f"Ящик с усами: {column}")
            plt.xlabel(column)
            file_path = os.path.join(self.output_dir, f"boxplot_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

        # Создаем сводную таблицу
        outlier_summary_df = pd.DataFrame(outlier_summary)
        return outlier_summary_df

    @timing
    def detect_outliers_zscore(self, numeric_columns, threshold=3):
        """
        Найти выбросы на основе Z-score для числовых переменных.

        Параметры:
        numeric_columns (list): Список числовых колонок.
        threshold (float): Пороговое значение Z-score для определения выбросов.

        Возвращает:
        pd.DataFrame: Сводная таблица с признаками и выбросами.
        """

        outlier_summary = []
        for column in numeric_columns:
            # Убираем пропуски для расчета Z-score
            non_na_data = self.df[column].dropna()
            z_scores = zscore(non_na_data)
            
            # Применяем фильтрацию, используя индексировку
            outliers = non_na_data[(z_scores > threshold) | (z_scores < -threshold)].tolist()

            # Добавляем данные о выбросах в сводную таблицу
            outlier_summary.append({
                'Признак': column,
                'Выбросы': outliers
            })

            # Построение графика выбросов (Boxplot для Z-score)
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[column], palette="viridis")
            plt.title(f"Выбросы на основе Z-score: {column}")
            plt.xlabel(column)

            # Сохранение графика
            file_path = os.path.join(self.output_dir, f"zscore_boxplot_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

        # Создаем сводную таблицу
        outlier_summary_df = pd.DataFrame(outlier_summary)
        return outlier_summary_df

    @timing
    def analyze_categorical_cross_tabulations(self, categorical_columns):
        """
        Построить одну таблицу сопряженности для всех категориальных переменных.
        Автоматически сделать выводы о взаимосвязях с учётом данных формата даты.

        Параметры:
        categorical_columns (list): Список категориальных колонок.

        Возвращает:
        pd.DataFrame: Одна объединённая таблица сопряженности.
        dict: Автоматические выводы о взаимосвязях.
        """
        combined_cross_tab = []
        conclusions = {}

        for i, col1 in enumerate(categorical_columns):
            # Удаляем пропуски
            col1_data = self.df[col1].dropna()

            # Пропускаем столбцы с датами
            try:
                parsed_dates = pd.to_datetime(col1_data, format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:
                    print(f"Пропущен анализ для столбца {col1}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {col1}: {e}")
                continue

            for col2 in categorical_columns[i + 1:]:
                # Удаляем пропуски
                col2_data = self.df[col2].dropna()

                # Пропускаем столбцы с датами
                try:
                    parsed_dates = pd.to_datetime(col2_data, format='%d.%m.%Y', errors='coerce')
                    if parsed_dates.notna().mean() > 0.8:
                        print(f"Пропущен анализ для столбца {col2}, так как он содержит данные формата даты.")
                        continue
                except Exception as e:
                    print(f"Ошибка при проверке формата даты для столбца {col2}: {e}")
                    continue

                # Таблица сопряженности
                ctab = pd.crosstab(self.df[col1], self.df[col2], dropna=True)

                # Добавляем информацию о паре переменных
                ctab.index = pd.MultiIndex.from_product([[col1], ctab.index], names=["Признак 1", "Категории признака 1"])
                ctab.columns = pd.MultiIndex.from_product([[col2], ctab.columns], names=["Признак 2", "Категории признака 2"])
                combined_cross_tab.append(ctab)

                # Анализ связи
                unique_combinations = (ctab.sum(axis=1) == 1).all() and (ctab.sum(axis=0) == 1).all()
                if unique_combinations:
                    conclusions[f"{col1} vs {col2}"] = "Чёткая взаимосвязь 1:1 (каждой категории одной переменной соответствует только одна категория другой переменной)."
                else:
                    conclusions[f"{col1} vs {col2}"] = "Существуют неоднозначные связи (одна категория соответствует нескольким категориям другой переменной)."

        # Объединяем все таблицы в одну
        if combined_cross_tab:
            combined_cross_tab_df = pd.concat(combined_cross_tab, axis=0)
            return combined_cross_tab_df, conclusions
        else:
            print("Нет подходящих категориальных переменных для анализа.")
            return pd.DataFrame(), conclusions

    @timing
    def plot_numeric_pairplot(self, numeric_columns=None):
        """
        Построить графики парных зависимостей для числовых признаков и сохранить в файл.

        Параметры:
        numeric_columns (list): Список числовых колонок. Если None, используются все числовые столбцы.
        """
        if numeric_columns is None:
            numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_columns:
            print("Нет числовых колонок для построения графиков парных зависимостей.")
            return

        # Построение pairplot
        pairplot_fig = sns.pairplot(self.df[numeric_columns], diag_kind="kde", corner=True)
        pairplot_fig.fig.suptitle("Графики парных зависимостей числовых признаков", y=1.02, fontsize=16)

        # Сохранение графика
        file_path = os.path.join(self.output_dir, "numeric_pairplot.jpg")
        pairplot_fig.savefig(file_path, bbox_inches='tight')
        plt.show()

        print(f"Графики парных зависимостей сохранены в: {file_path}")

    @timing
    def find_rare_categories(self, categorical_columns, threshold=0.05):
        """
        Выявить редкие категории в категориальных переменных.

        Параметры:
        categorical_columns (list): Список категориальных колонок.
        threshold (float): Порог для определения редких категорий (доля от общего числа).

        Возвращает:
        dict: Словарь с редкими категориями для каждой переменной.
        """
        rare_categories = {}

        for col in categorical_columns:
            # Удаляем пропуски
            non_na_data = self.df[col].dropna()

            # Проверяем, если колонка полностью пустая
            if non_na_data.empty:
                print(f"Пропущен анализ для столбца {col}, так как он пустой после удаления пропусков.")
                continue

            # Проверяем, является ли колонка числовой
            if pd.api.types.is_numeric_dtype(non_na_data):
                print(f"Пропущен анализ для столбца {col}, так как он содержит числовые данные.")
                continue

            # Проверяем, является ли колонка датой
            try:
                parsed_dates = pd.to_datetime(non_na_data, format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:  # Если более 80% значений интерпретируются как даты
                    print(f"Пропущен анализ для столбца {col}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {col}: {e}")
                continue

            # Подсчитываем доли значений и выявляем редкие категории
            value_counts = non_na_data.value_counts(normalize=True)
            rare_values = value_counts[value_counts < threshold].index.tolist()
            rare_categories[col] = rare_values

        return rare_categories

    @timing
    def analyze_correlations(self, target_column=None, threshold=0.5, correlation_types=None, include_phi=False, include_cramers_v=False):
        """
        Анализ корреляций между числовыми признаками и целевой переменной.
        Параметры:
        target_column (str): Название столбца с целевой переменной (может быть числовой или категориальной).
        threshold (float): Порог для включения коррелирующих пар (по модулю).
        correlation_types (list): Список типов корреляций для анализа. Доступные значения: 
                                'pearson', 'spearman', 'kendall'. По умолчанию все три.
        include_phi (bool): Включить расчет Phi-коэффициента для бинарных данных.
        include_cramers_v (bool): Включить расчет Cramér's V для категориальных данных.
        Возвращает:
        dict: Словарь с результатами анализа:
            - 'correlations': DataFrame с корреляциями между числовыми признаками.
            - 'anova': DataFrame с результатами теста ANOVA для категориального таргета.
            - 'cramers_v': DataFrame с Cramer's V для категориального таргета и категориальных признаков.
        """
        results = {}
        
        # Если пользователь не указал типы корреляций, используем все три по умолчанию
        if correlation_types is None:
            correlation_types = ['pearson', 'spearman', 'kendall']
        
        # Предварительная проверка: наличие целевой переменной
        if target_column is None or target_column not in self.df.columns:
            raise ValueError("Укажите корректное название целевой переменной.")
        
        # Убираем пропуски и очищаем данные
        self.df = self.df.dropna()
        
        # Разделение признаков по типам
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_column not in numeric_columns + categorical_columns:
            raise ValueError("Целевая переменная должна быть числовой или категориальной.")
        
        # Функция для определения уровня связи по шкале Чеддока
        def cheddock_scale(value):
            if value < 0.1:
                return "Очень слабая связь"
            elif 0.1 <= value < 0.3:
                return "Слабая связь"
            elif 0.3 <= value < 0.5:
                return "Умеренная связь"
            elif 0.5 <= value < 0.7:
                return "Заметная связь"
            elif 0.7 <= value < 0.9:
                return "Высокая связь"
            elif 0.9 <= value < 1.0:
                return "Весьма высокая связь"
            else:
                return "Идеальная связь"
        
        # Функция для интерпретации Cramér's V для категориальных данных
        def cramers_v_scale(value):
            if value <= 0.2:
                return "Слабая связь"
            elif 0.2 < value <= 0.6:
                return "Умеренная связь"
            elif value > 0.6:
                return "Сильная связь"
            else:
                return "Нет связи"
        
        # 1. Корреляция для числовых признаков
        if target_column in numeric_columns:
            correlations = {}
            if 'pearson' in correlation_types:
                correlations['pearson'] = self.df[numeric_columns].corr(method='pearson')
            if 'spearman' in correlation_types:
                correlations['spearman'] = self.df[numeric_columns].corr(method='spearman')
            if 'kendall' in correlation_types:
                correlations['kendall'] = self.df[numeric_columns].corr(method='kendall')
            
            # Список для хранения пар с высокой корреляцией
            correlated_pairs = []
            processed_pairs = set()
            
            for col1 in numeric_columns:
                for col2 in numeric_columns:
                    if col1 != col2 and (col1, col2) not in processed_pairs and (col2, col1) not in processed_pairs:
                        pair_data = {"Признак 1": col1, "Признак 2": col2}
                        
                        # Добавляем выбранные корреляции
                        for corr_type in correlation_types:
                            if corr_type in correlations:
                                corr_value = abs(correlations[corr_type].loc[col1, col2])
                                pair_data[f"Корреляция {corr_type.capitalize()}"] = round(corr_value, 2)
                                pair_data[f"Вывод по шкале Чеддока ({corr_type.capitalize()})"] = cheddock_scale(corr_value)
                        
                        # Рассчитываем Phi-коэффициент и Cramér's V только если указано
                        if include_phi or include_cramers_v:
                            contingency_table = pd.crosstab(self.df[col1].round(), self.df[col2].round())
                            chi2, _, _, _ = chi2_contingency(contingency_table)
                            n = contingency_table.sum().sum()
                            rows, cols = contingency_table.shape
                            
                            # Phi-коэффициент только для 2x2 таблиц
                            if rows == 2 and cols == 2 and include_phi:
                                phi_coefficient = np.sqrt(chi2 / n) if n > 0 else 0
                                pair_data["Phi-коэффициент"] = round(phi_coefficient, 2)
                            
                            # Cramér's V для любых таблиц
                            if include_cramers_v:
                                min_dim = min(rows - 1, cols - 1)
                                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else None
                                pair_data["Cramer's V"] = round(cramers_v, 2) if cramers_v is not None else None
                        
                        # Проверяем пороговое значение хотя бы для одной корреляции
                        if any(pair_data.get(f"Корреляция {corr_type.capitalize()}") >= threshold for corr_type in correlation_types):
                            correlated_pairs.append(pair_data)
                            processed_pairs.add((col1, col2))
            
            # DataFrame с корреляциями
            correlated_df = pd.DataFrame(correlated_pairs).drop_duplicates(subset=["Признак 1", "Признак 2"])
            
            # Сортируем результаты по выбранной корреляции (если указано)
            if len(correlation_types) > 0:
                primary_corr_type = correlation_types[0]  # Берем первую корреляцию из списка
                sort_column = f"Корреляция {primary_corr_type.capitalize()}"
                if sort_column in correlated_df.columns:
                    correlated_df = correlated_df.sort_values(by=sort_column, ascending=False)
            
            results['correlations'] = correlated_df
        
        # 2. ANOVA для категориального таргета
        elif target_column in categorical_columns:
            anova_results = []
            for col in numeric_columns:
                unique_groups = self.df[target_column].dropna().unique()
                if len(unique_groups) > 1:  # Проверка на достаточное количество групп
                    groups = [self.df[col][self.df[target_column] == group] for group in unique_groups]
                    # ANOVA тест
                    f_stat, p_value = f_oneway(*groups)
                    anova_results.append({
                        "Признак": col,
                        "F-статистика": round(f_stat, 2),
                        "P-значение": round(p_value, 4)
                    })
            results['anova'] = pd.DataFrame(anova_results)
            
            # 3. Cramér's V для категориальных признаков
            cramers_v_results = []
            for col in categorical_columns:
                if col != target_column:
                    contingency_table = pd.crosstab(self.df[col], self.df[target_column])
                    chi2, _, _, _ = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    rows, cols = contingency_table.shape
                    min_dim = min(rows - 1, cols - 1)
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else None
                    cramers_v_results.append({
                        "Признак": col,
                        "Cramer's V": round(cramers_v, 2) if cramers_v is not None else None,
                        "Интерпретация (Cramer's V)": cramers_v_scale(cramers_v) if cramers_v is not None else "Нет связи"
                    })
            results['cramers_v'] = pd.DataFrame(cramers_v_results)
        
        return results

    @timing
    def analyze_phik_correlations(self, threshold=0.5):
        """
        Анализ Phik корреляций между всеми признаками.

        Параметры:
        threshold (float): Порог для включения коррелирующих пар (по модулю).

        Возвращает:
        pd.DataFrame: Таблица с парами признаков, их Phik корреляцией и выводами.
        """

        # Убираем пропуски и очищаем данные
        self.df = self.df.dropna()

        # Вычисляем Phik корреляции
        phik_corr = self.df.phik_matrix(interval_cols=None)

        # Функция для интерпретации Phik-коэффициента
        def interpret_phik(value):
            if value <= 0.2:
                return "Слабая связь"
            elif 0.2 < value <= 0.4:
                return "Умеренная связь"
            elif 0.4 < value <= 0.6:
                return "Заметная связь"
            elif 0.6 < value <= 0.8:
                return "Высокая связь"
            else:
                return "Очень высокая связь"

        # Список для хранения пар с высокой корреляцией
        correlated_pairs = []
        processed_pairs = set()

        for col1 in phik_corr.columns:
            for col2 in phik_corr.index:
                if col1 != col2 and (col1, col2) not in processed_pairs and (col2, col1) not in processed_pairs:
                    phik_value = abs(phik_corr.loc[col1, col2])

                    if phik_value >= threshold:
                        correlated_pairs.append({
                            "Признак 1": col1,
                            "Признак 2": col2,
                            "Phik-коэффициент": round(phik_value, 2),
                            "Вывод": interpret_phik(phik_value)
                        })
                        processed_pairs.add((col1, col2))

        # Создаем DataFrame с результатами
        phik_correlations_df = pd.DataFrame(correlated_pairs).drop_duplicates(subset=["Признак 1", "Признак 2"])

        return phik_correlations_df

    @timing
    def analyze_datetime_attributes(self, datetime_column):
        """
        Анализ временных атрибутов записей (день, месяц, час, год) с построением гистограмм,
        KDE и наложением нормального распределения.

        Параметры:
        datetime_column (str): Название столбца с временными метками.

        Возвращает:
        dict: Словарь с DataFrame для каждого временного атрибута (годы, месяцы, дни, часы).
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])

        # Извлекаем атрибуты временных меток
        self.df['Год'] = self.df[datetime_column].dt.year
        self.df['Месяц'] = self.df[datetime_column].dt.month
        self.df['День'] = self.df[datetime_column].dt.day
        self.df['Час'] = self.df[datetime_column].dt.hour

        # Список временных атрибутов
        attributes = ['Год', 'Месяц', 'День', 'Час']
        summary_tables = {}

        for attr in attributes:
            data = self.df[attr].dropna()

            if data.nunique() <= 1:
                print(f"Пропущен график для '{attr}', так как значения одинаковы или отсутствует дисперсия.")
                continue

            plt.figure(figsize=(10, 6))

            # Гистограмма с KDE
            sns.histplot(
                data, 
                kde=True, 
                bins=30, 
                color='royalblue', 
                edgecolor='black', 
                alpha=0.8, 
                line_kws={"color": "orange", "linewidth": 2}, 
                label="KDE (плотность)"
            )

            # Вычисляем параметры нормального распределения
            mean = data.mean()
            std_dev = data.std()
            x_range = np.linspace(data.min(), data.max(), 1000)
            ideal_pdf = norm.pdf(x_range, loc=mean, scale=std_dev)
            ideal_pdf_scaled = ideal_pdf * len(data) * (data.max() - data.min()) / 30

            # Добавляем кривую нормального распределения
            plt.plot(
                x_range, 
                ideal_pdf_scaled, 
                color="red", 
                linestyle="--", 
                linewidth=2.5, 
                label="Идеальное распределение"
            )

            # Медиана и среднее
            median = data.median()
            plt.axvline(median, color="blue", linestyle="--", linewidth=2, label=f"Медиана: {median:.2f}")
            plt.axvline(mean, color="green", linestyle="-.", linewidth=2, label=f"Среднее: {mean:.2f}")

            # Добавление информации на график
            plt.title(f"Распределение по '{attr}'", fontsize=16)
            plt.xlabel(attr, fontsize=12)
            plt.ylabel("Частота", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Легенда
            plt.legend(fontsize=10)

            # Сохранение графика
            file_path = os.path.join(self.output_dir, f"datetime_attribute_distribution_{attr}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()

            print(f"График распределения по '{attr}' сохранён в: {file_path}")

            # Сводная таблица частот
            summary_table = data.value_counts().sort_index().reset_index()
            summary_table.columns = [attr, 'Частота']
            summary_tables[attr] = summary_table

        return summary_tables

    @timing
    def decompose_time_series(self, datetime_column, value_column):
        """
        Декомпозиция временного ряда на тренд, сезонность и остатки с выводом статистик.

        Параметры:
        datetime_column (str): Название столбца с временными метками.
        value_column (str): Название столбца с временными значениями.

        Возвращает:
        pd.DataFrame: Сводная таблица с основными статистиками для тренда, сезонности и остатков.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df = self.df.sort_values(by=datetime_column)
        ts = self.df.set_index(datetime_column)[value_column]

        stl = STL(ts, period=12)
        result = stl.fit()

        # Вывод графика декомпозиции
        fig = result.plot()
        fig.set_size_inches(10, 8)
        plt.suptitle(f"Декомпозиция временного ряда: {value_column}", fontsize=16)

        for ax in fig.axes:
            ax.tick_params(axis="x", rotation=45)

        file_path = os.path.join(self.output_dir, f"stl_decomposition_{value_column}.jpg")
        plt.savefig(file_path, bbox_inches='tight')
        plt.show()
        plt.close()

        # Расчет статистик
        components = {
            "Trend": result.trend,
            "Seasonal": result.seasonal,
            "Residual": result.resid
        }
        stats = {key: {
            "Среднее": comp.mean(),
            "Стандартное отклонение": comp.std(),
            "Минимум": comp.min(),
            "Максимум": comp.max()
        } for key, comp in components.items()}

        return pd.DataFrame(stats).T

    @timing
    def plot_autocorrelations(self, datetime_column, value_column, lags=50):
        """
        Построить графики автокорреляции (ACF) и частичной автокорреляции (PACF)
        и вернуть таблицу значений.

        Параметры:
        datetime_column (str): Название столбца с временными метками.
        value_column (str): Название столбца с временными значениями.
        lags (int): Количество лагов для анализа.

        Возвращает:
        pd.DataFrame: Таблица значений ACF и PACF.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df = self.df.sort_values(by=datetime_column)
        ts = self.df.set_index(datetime_column)[value_column]

        # Вычисление автокорреляции и частичной автокорреляции
        acf_values = acf(ts, nlags=lags, fft=True)
        pacf_values = pacf(ts, nlags=lags)

        # Построение графиков
        plt.figure(figsize=(12, 6))
        plot_acf(ts, lags=lags, title="ACF (Автокорреляция)")
        plt.savefig(os.path.join(self.output_dir, f"acf_{value_column}.jpg"), bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(12, 6))
        plot_pacf(ts, lags=lags, title="PACF (Частичная автокорреляция)")
        plt.savefig(os.path.join(self.output_dir, f"pacf_{value_column}.jpg"), bbox_inches='tight')
        plt.show()

        # Создание таблицы с ACF и PACF
        acf_pacf_table = pd.DataFrame({
            "Лаг": range(len(acf_values)),
            "ACF (Автокорреляция)": acf_values,
            "PACF (Частичная автокорреляция)": pacf_values
        })

        return acf_pacf_table

    @timing
    def check_stationarity(self, datetime_column, value_column):
        """
        Проверка стационарности временного ряда с использованием теста Дики-Фуллера.

        Параметры:
        datetime_column (str): Название столбца с временными метками.
        value_column (str): Название столбца с временными значениями.

        Возвращает:
        pd.DataFrame: Результаты теста Дики-Фуллера в табличной форме.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df = self.df.sort_values(by=datetime_column)
        ts = self.df.set_index(datetime_column)[value_column]

        result = adfuller(ts.dropna())
        stats = {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Critical Value (1%)": result[4]["1%"],
            "Critical Value (5%)": result[4]["5%"],
            "Critical Value (10%)": result[4]["10%"],
            "Stationary": result[1] < 0.05
        }
        return pd.DataFrame([stats])

    @timing
    def plot_seasonality_heatmap(self, datetime_column, value_column, freq='month'):
        """
        Построить тепловую карту сезонности и вернуть используемую таблицу.

        Параметры:
        datetime_column (str): Название столбца с временными метками.
        value_column (str): Название столбца с временными значениями.
        freq (str): Частота ("month", "day_of_week", "hour").

        Возвращает:
        pd.DataFrame: Таблица, используемая для построения тепловой карты.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df['Year'] = self.df[datetime_column].dt.year

        if freq == 'month':
            self.df['Month'] = self.df[datetime_column].dt.month
            pivot = self.df.pivot_table(index='Year', columns='Month', values=value_column, aggfunc='mean')
            ylabel, xlabel = "Год", "Месяц"
        elif freq == 'day_of_week':
            self.df['DayOfWeek'] = self.df[datetime_column].dt.dayofweek
            pivot = self.df.pivot_table(index='Year', columns='DayOfWeek', values=value_column, aggfunc='mean')
            ylabel, xlabel = "Год", "День недели"
        elif freq == 'hour':
            self.df['Hour'] = self.df[datetime_column].dt.hour
            pivot = self.df.pivot_table(index='Year', columns='Hour', values=value_column, aggfunc='mean')
            ylabel, xlabel = "Год", "Час"
        else:
            raise ValueError("Неверное значение freq. Используйте 'month', 'day_of_week' или 'hour'.")

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, cmap='viridis', annot=True, fmt=".1f", linewidths=0.5)
        plt.title(f"Тепловая карта сезонности: {value_column}", fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        file_path = os.path.join(self.output_dir, f"seasonality_heatmap_{value_column}.jpg")
        plt.savefig(file_path, bbox_inches='tight')
        plt.show()

        return pivot

    @timing
    def plot_time_series_with_table(self, datetime_column, value_column):
        """
        Построить график временного ряда и вернуть таблицу данных в формате pandas DataFrame.

        Параметры:
        datetime_column (str): Название столбца с временными метками.
        value_column (str): Название столбца с временными значениями.

        Возвращает:
        pd.DataFrame: Таблица данных, использованная для отображения.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df = self.df.sort_values(by=datetime_column)

        # Подготовка данных для таблицы (все строки)
        table_data = self.df[[datetime_column, value_column]].reset_index(drop=True)

        # Построение графика
        plt.figure(figsize=(12, 6))
        plt.plot(self.df[datetime_column], self.df[value_column], marker="o", linestyle="-", color="blue")
        plt.title(f"Временные ряды: {value_column}", fontsize=16)
        plt.xlabel("Дата", fontsize=12)
        plt.ylabel("Значение", fontsize=12)
        plt.grid(True)

        # Сохранение графика
        file_path = os.path.join(self.output_dir, f"time_series_{value_column}.jpg")
        plt.savefig(file_path, bbox_inches="tight")
        plt.show()

        print(f"График временного ряда сохранён в: {file_path}")

        # Возвращаем таблицу данных
        return table_data

    @timing
    def save_all_summaries_to_excel(self, summaries):
        """
        Сохранить все сводные таблицы в один Excel файл, где каждая таблица находится на отдельном листе.

        Параметры:
        summaries (dict): Словарь с именами листов в качестве ключей и DataFrame в качестве значений.
        """
        file_path = "DATA_OUT/eda_information.xlsx"
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            for sheet_name, df in summaries.items():
                # Ограничение длины имени листа до 31 символа
                valid_sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=valid_sheet_name, index=False)
        print(f"Все сводные таблицы сохранены в файл: {file_path}")