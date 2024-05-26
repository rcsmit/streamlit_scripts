
import sqlite3
import streamlit as st

def display_database_contents(db_path):
    """_summary_

    Args:
        db_path (_type_): _description_
    """    

    with sqlite3.connect(db_path) as conn:
        # Create a cursor object
        cursor = conn.cursor()
        
        # Fetch all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Iterate over each table and st.write its contents
        for table_name in tables:
            table_name = table_name[0]
            st.write(f"Contents of table {table_name}:")
            
            # Fetch all rows from the table
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
            rows = cursor.fetchall()
            
            # Fetch column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [description[1] for description in cursor.fetchall()]
            
            # Print column names
            st.write(f"Columns: {', '.join(columns)}")
            
            # Print rows
            for row in rows:
                st.write(row)
            
            st.write("\n" + "-"*50 + "\n")
        
  

def display_table_and_column_names(db_path):
    """_summary_

    Args:
        db_path (_type_): _description_
    """    
  
    with sqlite3.connect(db_path) as conn:
        # Create a cursor object
        cursor = conn.cursor()
        
        # Fetch all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Iterate over each table and st.write its name and columns
        for table_name in tables:
            table_name = table_name[0]
            st.write(f"Table: {table_name}")
            
            # Fetch column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [description[1] for description in cursor.fetchall()]
            
            # Print column names
            st.write("Columns:", ', '.join(columns))
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Print row count
            st.write(f"Number of rows: {row_count}")
            st.write("\n" + "-"*50 + "\n")
        
    
def delete_rows_except_first_100(db_path, table):
    """_summary_

    Args:
        db_path (_type_): _description_
        table (_type_): _description_
    """    
    # Connect to the database
    with sqlite3.connect(db_path) as conn:
        # Create a cursor object
        cursor = conn.cursor()
        
        # Fetch the IDs of the first 100 rows
        cursor.execute(f"SELECT rowid FROM {table} ORDER BY rowid LIMIT 100")
        rows_to_keep = cursor.fetchall()
        
        if rows_to_keep:
            # Convert the list of tuples to a list of IDs
            ids_to_keep = [row[0] for row in rows_to_keep]
            
            # Create a string of placeholders for the IN clause
            placeholders = ', '.join('?' for _ in ids_to_keep)
            
            # Delete rows not in the list of IDs to keep
            cursor.execute(f"DELETE FROM {table} WHERE rowid NOT IN ({placeholders})", ids_to_keep)
            
            # Commit the transaction
            conn.commit()
            st.write(f"Deleted all rows except the first 100 from '{table}'.")
        else:
            st.write("Table '{table}' is empty or has less than 100 rows. No rows deleted.")
        
       


def fetch_items_with_categorie(db_path, selected_dochter_id, selected_categories):
    """Show the items of the checklist

    Args:
        db_path (_type_): _description_
        selected_dochter_id (_type_): _description_
        selected_categories (_type_): _description_
    """
    # Connect to the database
    #conn = sqlite3.connect(db_path)
    with sqlite3.connect(db_path) as conn:
        # Create a cursor object
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM dochters where ID_dochter = '{selected_dochter_id}'")
        
        dochters_results = cursor.fetchall()
    
        for row in dochters_results:
            st.header(row[1])
            # st.write(row[5]) # meaning first crossbox
            # st.write(row[6]) # meaning second crossbox
            # st.write(row[7]) # meaning third crossbox
            st.write(row[10]) # source of the checklist
            
        st.write("\n" + "-"*50 + "\n")

        #  Define the SQL query with a join between items and Categorien
        placeholders = ",".join(["?" for _ in selected_categories])
        query = f"""
        SELECT items.id, Categorien.Categorie, items.item, items.toelichting, items.volgorde
        FROM items
        JOIN Categorien ON items.rel_categorie = Categorien.Id
        WHERE Categorien.rel_dochter = ? AND Categorien.Categorie IN ({placeholders})
    
        """
        # LIMIT 100 # add to query while debugging 

        # Execute the query
        cursor.execute(query, [selected_dochter_id] + selected_categories)
        
        # Fetch all results
        results = cursor.fetchall()
   
    
    # Organize results by categories
    categorized_items = {}
    for row in results:
        item_id, categorie, item, toelichting, volgorde = row
        if categorie not in categorized_items:
            categorized_items[categorie] = []
        categorized_items[categorie].append(item)
    
    # Print the results in the desired format
    for categorie, items in categorized_items.items():
        st.subheader(f"{categorie}")
        for item in items:
            st.write(f"O {item}")
        st.write()  # Add a blank line between categories

def fetch_categories(db_path, selected_dochter_id):
    """Fetch the categories for the multiselect box in the sidebar

    Args:
        db_path (_type_): _description_
        selected_dochter_id (_type_): _description_

    Returns:
        _type_: _description_
    """

    
    with sqlite3.connect(db_path) as conn:
        # Create a cursor object
        cursor = conn.cursor()

        # Fetch the distinct categories from the 'Categorien' table
        cursor.execute(f"SELECT DISTINCT Categorie FROM Categorien WHERE Categorien.rel_dochter = '{selected_dochter_id}'")
        
        categories_results = cursor.fetchall()
        
        

    # Extract category names
    categories = [row[0] for row in categories_results]
        
    return categories

def fetch_dochters_dropdown(db_path):
    """fetch the daughters for the main dropbox in the sidebar

    Args:
        db_path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Connect to the database
    with sqlite3.connect(db_path) as conn:
        # Create a cursor object
        cursor = conn.cursor()

        # Fetch the contents of the 'dochters' table
        cursor.execute("SELECT ID_dochter, dochternaam FROM dochters")
        dochters_results = cursor.fetchall()
        
      
    # Extract dochter IDs and names
    dochter_ids = [row[0] for row in dochters_results]
    dochter_names = [row[1] for row in dochters_results]

    # Create dropdown menu
    selected_dochter_id = st.sidebar.selectbox("Select a dochter", options=dochter_ids, format_func=lambda x: dochter_names[dochter_ids.index(x)])
    
    return selected_dochter_id


if __name__ == "__main__":
    db_path = r"input\yepcheck.sqlite"
    #display_database_contents(db_path)
    #display_table_and_column_names(db_path)
    #delete_rows_except_first_100(db_path, "mailinglist")
    selected_dochter_id = fetch_dochters_dropdown(db_path)
 
    # Fetch categories  # Display checkboxes in the sidebar
    categories = fetch_categories(db_path, selected_dochter_id)
    selected_categories = st.sidebar.multiselect("Select categories", categories, categories)
    if  len (selected_categories) == 0:
        st.error("Select at least one category")
        st.stop()
    else:
        # Show items from selected categories
        fetch_items_with_categorie(db_path, selected_dochter_id, selected_categories)
