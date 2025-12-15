import pygetwindow as gw


"""Prints the titles of all visible windows."""
try:
    # Get a list of all window titles
    window_titles = gw.getAllTitles()
    
    print("Currently open windows:")
    # Iterate over the list and print each title
    for title in window_titles:
        # Filter out empty or system-related titles for cleaner output
        if title and title not in ('', 'Windows Program Manager', 'Cortana'):
            print(f"- {title}")
except Exception as e:
    print(f"An error occurred: {e}")
    print("PyGetWindow may not support your operating system or setup well.")
    print("Consider trying 'PyWinCtl' as an alternative: pip install PyWinCtl")