import matplotlib.pyplot as plt
import pandas as pd

class CsvProcessor:

    '''Clase basica que permite una limpieza general e incluye las funciones estándar de carga y análisis'''
    
    def __init__(self, df):
        self.df = df
        print(f'Original shape: {self.df.shape}')

    @classmethod
    def load_data(cls, path:str, sep: str = ',', **kwargs):
        df = pd.read_csv(path, sep = sep, **kwargs)
        return cls(df)

    def missing_data(self):
        # Check for missing data
        missing_df = self.df.isnull().sum().rename('missing').reset_index()
        missing_df['missing_percentage'] = missing_df['missing']/self.df.shape[0]*100
        return missing_df[missing_df['missing'] >0].style.bar(subset='missing_percentage',color='red')
        
    def describe(self, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        """
        Devuelve estadísticas descriptivas para columnas numéricas.
        """
        return self.df.describe(percentiles=percentiles).T
        
    def clean(self):
        # Borrar columnas vacías, si las hay
        print('Cleaning dataset')
        print(f'Original shape: {self.df.shape}')
        data = self.df.copy()
        data = data.dropna(axis=1, how='all')
        
        print(f'Shape after cleaning: {self.df.shape}')
        self.df = data

    def print_columns(self):
        for col in self.df.columns:
            print(col)

    def get_data(self):
        return self.df.copy()

    def detect_outliers_iqr(self, multiplier=1.5):
        """
        Detecta outliers usando el método IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
        Devuelve un DataFrame con el conteo de outliers por variable numérica.
        """
        outlier_summary = []
    
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
    
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
    
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)].shape[0]
    
            outlier_summary.append({
                'variable': col,
                'outliers': outliers,
                'percentage': outliers / len(self.df) * 100
            })
    
        return pd.DataFrame(outlier_summary).sort_values('outliers', ascending=False)

    def get_outlier_rows(self, column, multiplier=1.5):
        """
        Devuelve las filas que contienen outliers en una columna dada
        usando el método IQR.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
    
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
    
        return self.df[(self.df[column] < lower) | (self.df[column] > upper)]

    
    def plot_distributions(self, bins=30):
        """
        Grafica histogramas de todas las variables numéricas.
        """
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        self.df[numeric_cols].hist(bins=bins, figsize=(14, 12))

    
    def correlation_matrix(self):
        """
        Devuelve la matriz de correlación entre variables numéricas.
        """
        return self.df.corr(numeric_only=True)

    def plot_correlation_matrix(self):
        """
        Grafica de la matriz de correlación usando matplotlib.
        """
        corr = self.df.corr(numeric_only=True)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(corr, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()


    def categorical_summary(self):
        """
        Devuelve número de categorías únicas por cada columna categórica.
        """
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
    
        return pd.DataFrame({
            'variable': cat_cols,
            'unique_values': [self.df[col].nunique() for col in cat_cols]
        })