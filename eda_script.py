import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from scipy.stats import kurtosis, skew
from scipy.stats import gaussian_kde

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

            # Строим график
            plt.figure(figsize=(8, 4))
            ax = sns.countplot(x=non_na_data.astype(str), palette="viridis")
            plt.title(f"Распределение категориальной переменной: {column}")
            plt.xlabel(column)
            plt.ylabel("Количество")
            plt.xticks(rotation=45)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
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

    def plot_categorical_vs_target(self, target_column, categorical_columns):
        """
        Построить графики сравнения категориальных переменных и целевой переменной и сохранить их в файлы.

        Параметры:
        target_column (str): Название колонки с целевой переменной.
        categorical_columns (list): Список категориальных колонок.
        """
        for column in categorical_columns:
            # Удаляем пропуски в текущем столбце и целевой переменной
            non_na_data = self.df[[column, target_column]].dropna()

            # Если после удаления пропусков данных нет, пропускаем
            if non_na_data.empty:
                print(f"Пропущен график для столбца {column}, так как он пустой после удаления пропусков.")
                continue

            # Проверяем формат данных: даты
            try:
                parsed_dates = pd.to_datetime(non_na_data[column], format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:  # Если более 80% значений интерпретируются как даты
                    print(f"Пропущен график для столбца {column}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {column}: {e}")
                continue

            # Проверяем тип данных: исключаем числовые столбцы
            if pd.api.types.is_numeric_dtype(non_na_data[column]):
                print(f"Пропущен график для столбца {column}, так как он содержит числовые данные.")
                continue

            # Строим график
            plt.figure(figsize=(8, 4))
            ax = sns.countplot(
                x=non_na_data[column].astype(str),
                hue=non_na_data[target_column].astype(str),
                palette="viridis"
            )
            plt.title(f"Распределение {column} в зависимости от {target_column}")
            plt.xlabel(column)
            plt.ylabel("Количество")
            plt.xticks(rotation=45)
            plt.legend(title=target_column)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
            file_path = os.path.join(self.output_dir, f"categorical_vs_target_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

    def detect_outliers_iqr(self, numeric_columns):
        """
        Построить графики "Ящик с усами" для числовых переменных и сохранить их в файлы, а также определить выбросы.

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

    def plot_kde_distributions(self, numeric_columns):
        """
        Построить гистограммы и графики KDE для всех числовых переменных с улучшенным оформлением и аналитикой.

        Параметры:
        numeric_columns (list): Список числовых колонок.

        Возвращает:
        pd.DataFrame: Итоговая сводная таблица с аналитикой по признакам.
        """
        from scipy.stats import kurtosis, skew

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
                color="royalblue",  # Основной цвет столбиков
                alpha=0.8,          # Прозрачность столбиков
                linewidth=1.2       # Толщина линии на краях столбиков
            )
            plt.title(f"Гистограмма и KDE для признака: {column}", fontsize=14, fontweight="bold")
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Плотность / Частота", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Аннотация медианы и среднего
            median = data.median()
            mean = data.mean()
            plt.axvline(median, color="red", linestyle="--", label=f"Медиана: {median:.2f}")
            plt.axvline(mean, color="blue", linestyle="-.", label=f"Среднее: {mean:.2f}")

            # Попробуем вычислить пиковое значение KDE
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 1000)  # Диапазон значений для KDE
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
                continue

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
                "Пик": round(peak_x, 2) if 'peak_x' in locals() else None,
                "Среднее": round(mean, 2),
                "Медиана": round(median, 2),
                "Распределение": range_info,
                "Вывод": distribution
            })

        # Создаем итоговую таблицу
        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def detect_outliers_zscore(self, numeric_columns, threshold=3):
        """
        Найти выбросы на основе Z-score для числовых переменных.

        Параметры:
        numeric_columns (list): Список числовых колонок.
        threshold (float): Пороговое значение Z-score для определения выбросов.

        Возвращает:
        pd.DataFrame: Сводная таблица с признаками и выбросами.
        """
        from scipy.stats import zscore

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

    def analyze_categorical_cross_tabulations(self, categorical_columns):
        """
        Построить одну таблицу сопряженности для всех категориальных переменных.
        Автоматически сделать выводы о взаимосвязях.

        Параметры:
        categorical_columns (list): Список категориальных колонок.

        Возвращает:
        pd.DataFrame: Одна объединённая таблица сопряженности.
        dict: Автоматические выводы о взаимосвязях.
        """
        combined_cross_tab = []
        conclusions = {}

        for i, col1 in enumerate(categorical_columns):
            for col2 in categorical_columns[i + 1:]:
                # Таблица сопряженности
                ctab = pd.crosstab(self.df[col1], self.df[col2], dropna=True)
                
                # Добавляем информацию о паре переменных
                ctab.index = pd.MultiIndex.from_product([[col1], ctab.index], names=["Variable1", "Category1"])
                ctab.columns = pd.MultiIndex.from_product([[col2], ctab.columns], names=["Variable2", "Category2"])
                combined_cross_tab.append(ctab)

                # Анализ связи
                unique_combinations = (ctab.sum(axis=1) == 1).all() and (ctab.sum(axis=0) == 1).all()
                if unique_combinations:
                    conclusions[f"{col1} vs {col2}"] = "Чёткая взаимосвязь 1:1 (каждой категории одной переменной соответствует только одна категория другой переменной)."
                else:
                    conclusions[f"{col1} vs {col2}"] = "Существуют неоднозначные связи (одна категория соответствует нескольким категориям другой переменной)."

        # Объединяем все таблицы в одну
        combined_cross_tab_df = pd.concat(combined_cross_tab, axis=0)
        return combined_cross_tab_df, conclusions


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
            value_counts = self.df[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < threshold].index.tolist()
            rare_categories[col] = rare_values
        return rare_categories

    def analyze_correlations(self, target_column=None, threshold=0.5):
        """
        Анализ корреляций между числовыми признаками и целевой переменной.

        Параметры:
        target_column (str): Название столбца с целевой переменной (может быть числовой или категориальной).
        threshold (float): Порог для включения коррелирующих пар (по модулю).

        Возвращает:
        dict: Словарь с результатами анализа:
            - 'correlations': DataFrame с корреляциями между числовыми признаками.
            - 'anova': DataFrame с результатами теста ANOVA для категориального таргета.
            - 'cramers_v': DataFrame с Cramer's V для категориального таргета и категориальных признаков.
        """
        from scipy.stats import f_oneway, chi2_contingency
        import numpy as np

        results = {}

        # Проверка на наличие целевой переменной
        if target_column is None or target_column not in self.df.columns:
            raise ValueError("Укажите корректное название целевой переменной.")

        # Разделение признаков по типам
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Убираем пропуски
        cleaned_df = self.df.dropna()

        # 1. Корреляция для числовых признаков
        if target_column in numeric_columns:
            pearson_corr = cleaned_df.corr(method='pearson')
            spearman_corr = cleaned_df.corr(method='spearman')
            kendall_corr = cleaned_df.corr(method='kendall')

            # Список для хранения пар с высокой корреляцией
            correlated_pairs = []
            processed_pairs = set()

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

            for col1 in numeric_columns:
                for col2 in numeric_columns:
                    if col1 != col2 and (col1, col2) not in processed_pairs and (col2, col1) not in processed_pairs:
                        pearson_value = abs(pearson_corr.loc[col1, col2])
                        spearman_value = abs(spearman_corr.loc[col1, col2])
                        kendall_value = abs(kendall_corr.loc[col1, col2])

                        if (pearson_value >= threshold or
                            spearman_value >= threshold or
                            kendall_value >= threshold):

                            # Рассчитываем среднее значение корреляций
                            mean_correlation = (pearson_value + spearman_value + kendall_value) / 3
                            final_cheddock = cheddock_scale(mean_correlation)

                            correlated_pairs.append({
                                "Признак 1": col1,
                                "Признак 2": col2,
                                "Корреляция Пирсона": round(pearson_value * 100, 2),
                                "Вывод по шкале Чеддока (Пирсон)": cheddock_scale(pearson_value),
                                "Корреляция Спирмена": round(spearman_value * 100, 2),
                                "Вывод по шкале Чеддока (Спирмен)": cheddock_scale(spearman_value),
                                "Корреляция Кендала": round(kendall_value * 100, 2),
                                "Вывод по шкале Чеддока (Кендал)": cheddock_scale(kendall_value),
                                "Средняя корреляция": round(mean_correlation * 100, 2),
                                "Итоговый вывод по шкале Чеддока": final_cheddock
                            })
                            processed_pairs.add((col1, col2))

            # DataFrame с корреляциями
            correlated_df = pd.DataFrame(correlated_pairs).drop_duplicates(subset=["Признак 1", "Признак 2"])
            results['correlations'] = correlated_df

        # 2. ANOVA для категориального таргета
        elif target_column in categorical_columns:
            anova_results = []

            for col in numeric_columns:
                unique_groups = cleaned_df[target_column].dropna().unique()
                if len(unique_groups) > 1:  # Проверка на достаточное количество групп
                    groups = [cleaned_df[col][cleaned_df[target_column] == group] for group in unique_groups]

                    # ANOVA тест
                    f_stat, p_value = f_oneway(*groups)
                    anova_results.append({
                        "Признак": col,
                        "F-статистика": round(f_stat, 2),
                        "P-значение": round(p_value, 4)
                    })

            results['anova'] = pd.DataFrame(anova_results)

            # 3. Критерий Крамера V для категориальных признаков
            cramers_v_results = []

            for col in categorical_columns:
                if col != target_column:
                    contingency_table = pd.crosstab(cleaned_df[col], cleaned_df[target_column])
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

                    cramers_v_results.append({
                        "Признак": col,
                        "Cramer\'s V": round(cramers_v, 2)
                    })

            results['cramers_v'] = pd.DataFrame(cramers_v_results)

        else:
            raise ValueError("Тип целевой переменной не распознан. Она должна быть числовой или категориальной.")

        return results

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