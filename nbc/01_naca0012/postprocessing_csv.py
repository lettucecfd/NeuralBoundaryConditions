import sys
import pandas as pd
import re

csv_file = sys.argv[1]
output_file = sys.argv[2]
U = None
Rho = 1.0

if len(sys.argv) >= 4:
    U = float(sys.argv[3])
if len(sys.argv) == 5:
    Rho = float(sys.argv[4])

try:
    data = pd.read_csv(csv_file)

    # If U is not provided, read from the second row of the column 'Velocity_x-inlet'
    if U is None:
        if 'Velocity_x-inlet' in data.columns:
            U = data['Velocity_x-inlet'].iloc[1]
            print(f"U value extracted from 'Velocity_x-inlet': {U}")
        else:
            print("'Velocity_x-inlet' not found in column names.")
            sys.exit(1)

    # Find column index with entry 'Ax'
    col_index_ax = data.columns.get_loc('Ax') if 'Ax' in data.columns else None
    if col_index_ax is not None:
        print(f"The column index for 'Ax' is: {col_index_ax}")
        # Store the first entry (row 2) in variable Ax
        Ax = data.iloc[1, col_index_ax]
        print(f"The first entry in column 'Ax' (row 2) is: {Ax}")
    else:
        print("'Ax' not found in column names.")

    # Find column index with entry 'Ay'
    col_index_ay = data.columns.get_loc('Ay') if 'Ay' in data.columns else None
    if col_index_ay is not None:
        print(f"The column index for 'Ay' is: {col_index_ay}")
        # Store the first entry (row 2) in variable Ay
        Ay = data.iloc[1, col_index_ay]
        print(f"The first entry in column 'Ay' (row 2) is: {Ay}")
    else:
        print("'Ay' not found in column names.")

    # Find column index with entry 'Drag'
    col_index_drag = data.columns.get_loc('Drag') if 'Drag' in data.columns else None
    if col_index_drag is not None:
        print(f"The column index for 'Drag' is: {col_index_drag}")
        # Extract all entries from column 'Drag'
        drag_values = data['Drag'].tolist()
        # print(f"All entries in column 'Drag': {drag_values}")
    else:
        print("'Drag' not found in column names.")

    # Find column index with entry 'Lift'
    col_index_lift = data.columns.get_loc('Lift') if 'Lift' in data.columns else None
    if col_index_lift is not None:
        print(f"The column index for 'Lift' is: {col_index_lift}")
        # Extract all entries from column 'Lift'
        lift_values = data['Lift'].tolist()
        # print(f"All entries in column 'Lift': {lift_values}")
    else:
        print("'Lift' not found in column names.")

    # Find column index with entry 'Time_si'
    col_index_time_si = data.columns.get_loc('Time_si') if 'Time_si' in data.columns else None
    if col_index_time_si is not None:
        print(f"The column index for 'Time_si' is: {col_index_time_si}")
        # Extract all entries from column 'Time_si'
        time_si_values = data['Time_si'].tolist()
        # print(f"All entries in column 'Time_si': {time_si_values}")
    else:
        print("'Time_si' not found in column names.")

    # Calculate force coefficients Cd and Cl
    if col_index_drag is not None and col_index_lift is not None and col_index_ax is not None and col_index_ay is not None:
        Ayy = re.search(r'GPD(\d+)', csv_file)
        if Ayy:
            gpd_value = int(Ayy.group(1))  # Zahl extrahieren und in eine ganze Zahl umwandeln
            print(f"Die Zahl hinter GPD ist: {gpd_value}")
        else:
            print("Keine Zahl hinter GPD gefunden.")
        cd_values = [(drag * 2) / (U ** 2 * Rho * gpd_value) for drag in drag_values]
        cl_values = [(lift * 2) / (U ** 2 * Rho * gpd_value) for lift in lift_values]
        # print(f"Calculated drag coefficients (Cd): {cd_values}")
        # print(f"Calculated lift coefficients (Cl): {cl_values}")

        # Create a new DataFrame for output
        output_data = pd.DataFrame({
            'Time': time_si_values,
            'Cd': cd_values,
            'Cl': cl_values
        })

        # Save the output DataFrame to a new CSV file
        output_data.to_csv(output_file, index=False)
        print(f"Output saved to {output_file}")

    # Print U and Rho values
    print(f"Value of U: {U}")
    print(f"Value of Rho: {Rho}")

except FileNotFoundError:
    print(f"Error: File '{csv_file}' not found.")
except pd.errors.EmptyDataError:
    print(f"Error: File '{csv_file}' is empty.")
except pd.errors.ParserError:
    print(f"Error: File '{csv_file}' could not be parsed.")
