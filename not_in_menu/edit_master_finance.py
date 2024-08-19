import openpyxl
from datetime import datetime
# Load the workbook
workbook = openpyxl.load_workbook('C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023.xlsx')
# Save the workbook
workbook.save(f'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx')

# Get the sheets
invoer_sheet = workbook['INVOER']
rules_sheet = workbook['RULES']

# Read rules into a dictionary
rules = {row[1].lower(): row[2:6] for row in rules_sheet.iter_rows(min_row=2, values_only=True) if row[1]}

# Process INVOER sheet
for row in invoer_sheet.iter_rows(min_row=2, min_col=8, max_col=19):
    description_invoer = row[3].value
    amount_invoer = row[0].value

    if description_invoer:
        description_invoer_lower = description_invoer.lower()
        for term, values in rules.items():
            amount_rules = values[3]

            # Check if term is in description and amount matches (if specified)
            if term in description_invoer_lower and (amount_rules is None or amount_invoer == amount_rules):
                print(f"MATCH {description_invoer}")

                # Only apply rules if target cells are empty
                if all(cell.value is None for cell in (row[4], row[5], row[6])):
                    row[4].value = values[0]  # income_expenses
                    row[5].value = values[1]  # main_category
                    row[6].value = values[2]  # category
                    row[11].value = f"generated {datetime.now().strftime('%Y-%m-%d')}"
                break  # Stop after the first match

# Save the workbook
workbook.save(f'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx')
workbook.save('C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023.xlsx')
