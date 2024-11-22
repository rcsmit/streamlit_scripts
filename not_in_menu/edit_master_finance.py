import openpyxl
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

def load_workbook(file_path: str) -> Optional[Workbook]:
    """
    Load an Excel workbook from the given file path.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        Optional[Workbook]: The loaded workbook, or None if the file was not found.
    """
    try:
        return openpyxl.load_workbook(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def save_backup(workbook: Workbook, backup_path: str) -> None:
    """
    Save a backup of the given workbook to the specified path.

    Args:
        workbook (Workbook): The workbook to save.
        backup_path (str): The path to save the backup file.
    """
    try:
        workbook.save(backup_path)
    except Exception as e:
        print(f"Error saving backup: {e}")

def get_sheets(workbook: Workbook) -> Tuple[Optional[Worksheet], Optional[Worksheet]]:
    """
    Get the 'INVOER' and 'RULES' sheets from the workbook.

    Args:
        workbook (Workbook): The workbook to get the sheets from.

    Returns:
        Tuple[Optional[Worksheet], Optional[Worksheet]]: The 'INVOER' and 'RULES' sheets, or (None, None) if not found.
    """
    try:
        invoer_sheet = workbook['INVOER']
        rules_sheet = workbook['RULES']
        return invoer_sheet, rules_sheet
    except KeyError as e:
        print(f"Error: Sheet {e} not found in the workbook.")
        return None, None

def read_rules(rules_sheet: Worksheet) -> Dict[str, Tuple[Any, Any, Any, Any]]:
    """
    Read rules from the 'RULES' sheet.

    Args:
        rules_sheet (Worksheet): The 'RULES' sheet to read from.

    Returns:
        Dict[str, Tuple[Any, Any, Any, Any]]: A dictionary with rule descriptions as keys and rule details as values.
    """
    return {row[1].lower(): row[2:6] for row in rules_sheet.iter_rows(min_row=2, values_only=True) if row[1]}

def process_invoer_sheet(invoer_sheet: Worksheet, rules: Dict[str, Tuple[Any, Any, Any, Any]]) -> None:
    """
    Process the 'INVOER' sheet using the provided rules.

    Args:
        invoer_sheet (Worksheet): The 'INVOER' sheet to process.
        rules (Dict[str, Tuple[Any, Any, Any, Any]]): The rules to apply to the 'INVOER' sheet.
    """
    for row in invoer_sheet.iter_rows(min_row=2, min_col=8, max_col=19):
        description_invoer = row[3].value
        amount_invoer = row[0].value

        if description_invoer:
            description_invoer_lower = description_invoer.lower()
            for term, values in rules.items():
                amount_rules = values[3]

                # Check if term is in description and amount matches (if specified)
                if term in description_invoer_lower and (amount_invoer == amount_rules):
                    print(f"MATCH {description_invoer}")

                    # Only apply rules if target cells are empty
                    if all(cell.value is None for cell in (row[4], row[5], row[6])):
                        row[4].value = values[0]  # income_expenses
                        row[5].value = values[1]  # main_category
                        row[6].value = values[2]  # category
                        row[11].value = f"generated {datetime.now().strftime('%Y-%m-%d')}"
                    break  # Stop after the first match

def main() -> None:
    """
    Main function to load the workbook, save a backup, and process the sheets.
    """
    file_path = 'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023.xlsx'
    backup_path = f'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

    workbook = load_workbook(file_path)
    if not workbook:
        return

    save_backup(workbook, backup_path)

    invoer_sheet, rules_sheet = get_sheets(workbook)
    if not invoer_sheet or not rules_sheet:
        return

    rules = read_rules(rules_sheet)
    process_invoer_sheet(invoer_sheet, rules)

    workbook.save(f'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023_updated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    workbook.save('C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023.xlsx')

    print("Processing completed.")

if __name__ == "__main__":
    main()
