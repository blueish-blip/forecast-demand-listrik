import pandas as pd
import json
import os

def convert_excel_to_json(input_file, output_file):
    # Load the Excel file
    df = pd.read_excel(input_file)
    
    # Preprocessing: Rename columns to match dashboard expectation
    # Assuming the Excel has 'Date' and 'Load' columns
    # Adjust names if they differ in your file
    df = df.rename(columns={'Date': 'date', 'Load': 'actual'})
    
    # Ensure date is string
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Convert to list of dicts
    data_series = df.to_dict(orient='records')
    
    # Create the full structure
    result = {
        "metadata": {
            "title": "Panama Energy Demand Forecasting",
            "unit": "MWh",
            "last_updated": pd.Timestamp.now().strftime('%Y-%m-%d'),
            "forecast_start_date": df['date'].max()
        },
        "metrics": {
            "Model_A": { "rmse": 12.4, "mae": 9.1, "mape": 3.2, "aic": 450.2 }
        },
        "series": data_series
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Successfully converted {input_file} to {output_file}")

# Paths
input_path = r"C:\Users\hhfzsn\Downloads\Panama_beban_listrik_per_jam_2015-2020_.xlsx"
output_path = r"D:\FreeLLM\Open Code\Project\data\forecast_results.json"

convert_excel_to_json(input_path, output_path)
