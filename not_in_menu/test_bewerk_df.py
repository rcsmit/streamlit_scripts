import pandas as pd

def get_data():
   
    maxxton_file = r"C:\Users\rcxsm\Downloads\ReservationDetails.xlsx"
    
    df_maxxton = pd.read_excel(maxxton_file)
    return df_maxxton

def compare_files(data_maxxton):
    """Compare the maxxton file with the data in Maxxton.
    """    
   
    data_maxxton = data_maxxton.copy()

    # Filter the DataFrame to remove rows where accommodation type starts with 'pitch'
    data_maxxton = data_maxxton[~data_maxxton["Accommodation Type"].str.startswith("Pitch")]
    data_maxxton["Reservation Number"] = (
        data_maxxton["Reservation Number"].astype(str).str[3:8].astype(int)
    )
    data_maxxton["Arrival Date"] = pd.to_datetime(
        data_maxxton["Arrival Date"], format="%d/%m/%Y"
    )
   
    # Mapping of original values to replacement values
    value_map_acc = {
        "Safari tent Serengeti XL Glamping Soleil": "SERENGETI XL",
        "Safari tent Kalahari Soleil [5 pers. (5 adults) 32m²": "KALAHARI1",
        "Safari tent Serengeti Glamping Soleil": "SERENGETI L",
        "Mobile home Waikiki Soleil": "WAIKIKI",
        "Safari tent Kalahari Soleil [5 pers. (5 adults) 25m²": "KALAHARI2",
        "Mobile home Bali Soleil": "BALI",
        "Bungalow tent Navajo Soleil": "SAHARA",
    }

    # Replace values in the 'Property' column
    for original_value, replacement_value in value_map_acc.items():
        data_maxxton.loc[
            data_maxxton["Accommodation Type"].str.startswith(original_value), "Accommodation Type",] = replacement_value
    try:
        value_map_country = {
            "Belgium":"be",
            "Switserland":"ch",
            "Germany":"de",
            "Denmark":"dk",
            "France":"fr",
            "Great Britain":"uk",
            "Luxembourg":"lx",
            "Netherlands":"nl",
            "Philippines (the)":"ph"}
        for original_value, replacement_value in value_map_country.items():
            data_maxxton.loc[
                data_maxxton["Country"].str.startswith(original_value), "Country",] = replacement_value.upper()
    except Exception:
        pass
    data_maxxton = data_maxxton[["Last Name","Reservation Number", "Arrival Date", "Country"]]
    print (data_maxxton)
def main():
    df_maxxton = get_data()
    compare_files(df_maxxton)

main()