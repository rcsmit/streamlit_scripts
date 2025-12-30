import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

def main():
    # Set page config
    try:
        st.set_page_config(page_title="Year Timeline Generator", layout="centered")
    except:
        pass  # Page config can only be set once

    st.title("📅 How This Year Went Generator")
    st.write("Adjust the sliders to indicate the subjective lenght.")

    # --- Sidebar Controls ---
    st.sidebar.header("Month Length Settings")
    st.sidebar.write("Small numbers = fast, Big numbers = Slow")

    months_default = {
        "January": 2, "February": 2, "March": 2,
        "April": 4, "May": 6, "June": 6,
        "July": 7, "August": 9, "September": 14,
        "October": 4, "November": 2, "December": 1
    }

    month_values = {}
    for month, default in months_default.items():
        month_values[month] = st.sidebar.slider(f"{month}", 1, 20, default)

    chart_title = st.sidebar.text_input("Title of the graph", "HOW THIS YEAR WENT:")

    # --- Plotting Function ---
    def create_timeline(data, title):
        # Use xkcd style for the hand-drawn look
        with plt.xkcd():
            # Dynamic height based on total units to keep aspect ratio decent
            total_units = sum(data.values())
            fig_height = max(8, total_units * 0.3)
            
            fig, ax = plt.subplots(figsize=(6, fig_height))
            
            # Calculate positions
            labels = list(data.keys())
            durations = list(data.values())
            y_positions = [0] + list(np.cumsum(durations))
            total_height = y_positions[-1]
            
            # Draw main vertical line
            ax.plot([1, 1], [0, total_height], color='k', lw=2.5)
            
            # Draw ticks and labels
            for i in range(len(labels)):
                y_top = y_positions[i]
                y_bot = y_positions[i+1]
                y_center = (y_top + y_bot) / 2
                
                # Horizontal Tick
                ax.plot([0.5, 1.5], [y_top, y_top], color='k', lw=2)
                
                # Label
                # Adjust font size slightly for very small intervals
                fs = 18
                if durations[i] <= 1:
                    fs = 10
                elif durations[i] <= 2:
                    fs = 14
                    
                ax.text(2, y_center, labels[i].upper(), fontsize=fs, va='center')#, fontname='DejaVu Sans')

            # Bottom Tick
            ax.plot([0.5, 1.5], [total_height, total_height], color='k', lw=2)

            # Title
            ax.text(0.5, -total_height * 0.05, title, fontsize=22, ha='left', weight='bold')

            # Formatting
            ax.set_ylim(total_height + (total_height*0.1), -(total_height*0.15))
            ax.set_xlim(0, 10)
            ax.axis('off')
            
            return fig

    # --- Generate and Display ---
    fig = create_timeline(month_values, chart_title)
    st.pyplot(fig)

    # --- Download Button ---
    fn = "my_year_timeline.png"
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')

    st.download_button(
        label="📥 Download Image",
        data=img,
        file_name=fn,
        mime="image/png"
    )

if __name__ == "__main__":
    main()