from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(data):
    # Define boolean columns
    bool_columns = ['is_student', 'has_license']
    
    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), []),
            ('cat', OneHotEncoder(), []),
            ('bool_str', FunctionTransformer(func=lambda x: x.astype(str)), bool_columns)
        ]
    )
    
    # Apply the preprocessing pipeline
    processed_data = preprocessor.fit_transform(data)
    
    return processed_data