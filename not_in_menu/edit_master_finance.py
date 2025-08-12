import openpyxl
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import pandas as pd

def load_workbook(file_path: str) -> Optional[Workbook]:
    """
    Load an Excel workbook from the given file path.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        Optional[Workbook]: The loaded workbook, or None if the file was not found.
    """
    print (f"Opening {file_path}")
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
    print (f"Saving backup to {backup_path}")
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
    print("Getting sheets 'INVOER' and 'RULES'")
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
    print("Reading rules from 'RULES' sheet")
    return {row[1].lower(): row[2:6] for row in rules_sheet.iter_rows(min_row=2, values_only=True) if row[1]}

def process_invoer_sheet(invoer_sheet: Worksheet, rules: Dict[str, Tuple[Any, Any, Any, Any]]) -> None:
    """
    Process the 'INVOER' sheet using the provided rules.

    Args:
        invoer_sheet (Worksheet): The 'INVOER' sheet to process.
        rules (Dict[str, Tuple[Any, Any, Any, Any]]): The rules to apply to the 'INVOER' sheet.
    """
    print ("Processing 'INVOER' sheet with rules")
    for row in invoer_sheet.iter_rows(min_row=2, min_col=8, max_col=20):
        
        description_invoer = row[4].value
        amount_invoer = row[0].value

        if description_invoer:
            try:
                description_invoer_lower = description_invoer.lower()
            except:
                description_invoer_lower = "_"
            for term, values in rules.items():
                amount_rules = values[3]

                # Check if term is in description and amount matches (if specified)
                if term in description_invoer_lower and (not amount_rules or amount_invoer == amount_rules):
                    print(f"MATCH {description_invoer}")

                    # Only apply rules if target cells are empty
                    if all(cell.value is None for cell in (row[5], row[6], row[7])):
                        row[5].value = values[0]  # income_expenses
                        row[6].value = values[1]  # main_category
                        row[7].value = values[2]  # category
                        row[12].value = f"generated {datetime.now().strftime('%Y-%m-%d')}"
                    break  # Stop after the first match


def main() -> None:
    """
    Main function to load the workbook, save a backup, and process the sheets.
    """
    file_path = 'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023.xlsx'
    backup_path = f'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    time_start = datetime.now().strftime("%H:%M:%S")
    print("Starting workbook load at " + time_start)

  
    workbook = load_workbook(file_path)
    if not workbook:
        print("Failed to load workbook. Exiting.")
        return
    time_end = datetime.now().strftime("%H:%M:%S")
    print('Load finished at ' + time_end)
  
    save_backup(workbook, backup_path)

    invoer_sheet, rules_sheet = get_sheets(workbook)
    if not invoer_sheet or not rules_sheet:
        print("Required sheets not found. Exiting.")
        return

    rules = read_rules(rules_sheet)
    process_invoer_sheet(invoer_sheet, rules)

    workbook.save(f'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023_updated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    workbook.save('C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023.xlsx')

    time_end = datetime.now().strftime("%H:%M:%S")
    print(f"Processing completed at {time_end}.")
    
    print(f"Total time taken: {datetime.strptime(time_end, '%H:%M:%S') - datetime.strptime(time_start, '%H:%M:%S')}")

def find_kruis_posten():
    """
    Placeholder function for finding 'kruis posten'.
    Currently does nothing but can be implemented as needed.
    """
    print("Finding 'kruis posten'... ")
    # # Implementation goes here
    # pass

    # file_path = 'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023.xlsx'
    # backup_path = f'C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    # time_start = datetime.now().strftime("%H:%M:%S")
    # print("Starting workbook load at " + time_start)

  
    # workbook = load_workbook(file_path)
    # if not workbook:
    #     print("Failed to load workbook. Exiting.")
    #     return
    # time_end = datetime.now().strftime("%H:%M:%S")
    # print('Load finished at ' + time_end)
  
    # save_backup(workbook, backup_path)

    # invoer_sheet = workbook['INVOER']

    
   
    # Lees
    path = r"C:\Users\rcxsm\Documents\xls\masterfinance_2023.xlsx"
    sheet = "INVOER"
    df = pd.read_excel(path, sheet_name=sheet)
        
    for jaar in [2020,2021,2022,2023,2024,2025]:
        print (f"Processing {jaar}")
        out_sheet = f"KRUIS_UNMATCHED_{jaar}"
        
        # Filter
        m = (df["year"] == jaar) & (df["category"].astype(str).str.upper() == "KRUIS")
        kruis = df.loc[m].copy()

        # Normaliseer bedrag op 2 decimals
        kruis["bedrag_norm"] = kruis["Bedrag"].round(2)
        kruis["abs_key"] = kruis["bedrag_norm"].abs()

        # Markeer matches per absolute bedrag
        kruis["matched"] = False

        for key, sub in kruis.groupby("abs_key"):
            pos_idx = sub.index[sub["bedrag_norm"] > 0].tolist()
            neg_idx = sub.index[sub["bedrag_norm"] < 0].tolist()
            k = min(len(pos_idx), len(neg_idx))
            # eerste k uit beide lijsten vormen een paar
            kruis.loc[pos_idx[:k], "matched"] = True
            kruis.loc[neg_idx[:k], "matched"] = True

        unmatched = kruis.loc[~kruis["matched"]].drop(columns=["bedrag_norm","abs_key","matched"])

        # Toon resultaat
        print(f"Aantal kruisposten zonder tegenhanger: {len(unmatched)}")
        print(unmatched[["date","Bedrag","Description","main_category","category"]].sort_values("date"))
        if len(unmatched)>0:
            # Schrijf naar nieuw tabblad
            with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
                unmatched.to_excel(xw, sheet_name=out_sheet, index=False)
        else:
            print(f"Nothing found in {jaar}")
if __name__ == "__main__":
    #main()
    find_kruis_posten()
