import streamlit as st

st.title("Simplified Content Viewer")

if st.button('Fetch Simplified Content'):
    # Assume you have a file path to your HTML file
    html_file_path = "generated_content.html"  # Replace with your actual file path

    # Function to load and display HTML content
    def load_and_display_html(file_path):
        with open(file_path, 'r') as f:
            html_content = f.read()
        return html_content

    # Load HTML content
    html_content = load_and_display_html(html_file_path)

    # Display HTML content using st.components or st.write
    # Adjust height as needed
    st.components.v1.html(html_content, height=5000)

    # Alternatively, use st.write to display HTML
    # st.write(html_content)
