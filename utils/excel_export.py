import pandas as pd
from datetime import datetime

def create_excel_report(data, filename=None):
    """
    Create Excel report with analysis results
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report_{timestamp}.xlsx"
    
    # Create Excel writer
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Write data to Excel
    data.to_excel(writer, sheet_name='Analysis Results', index=False)
    
    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Analysis Results']
    
    # Add formatting
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BC',
        'border': 1
    })
    
    # Write headers with formatting
    for col_num, value in enumerate(data.columns.values):
        worksheet.write(0, col_num, value, header_format)
    
    # Adjust column widths
    for i, col in enumerate(data.columns):
        max_length = max(
            data[col].astype(str).apply(len).max(),
            len(col)
        )
        worksheet.set_column(i, i, max_length + 2)
    
    # Save the Excel file
    writer.close()
    
    return filename

def append_to_excel(data, filename):
    """
    Append new data to existing Excel file
    """
    try:
        # Read existing Excel file
        existing_data = pd.read_excel(filename)
        
        # Concatenate new data
        combined_data = pd.concat([existing_data, data], ignore_index=True)
        
        # Save updated data
        return create_excel_report(combined_data, filename)
    
    except FileNotFoundError:
        # If file doesn't exist, create new one
        return create_excel_report(data, filename) 