import streamlit as st


st.title("About Us")

st.header("Our Mission")
st.write("""
    At Aegis, our mission is to provide top-notch cybersecurity solutions to protect your digital assets.
    We are committed to ensuring the safety and security of your information through innovative technologies and expert knowledge.
""")

st.header("Our Team")

col2, col3, col1 = st.columns(3)

with col1:
    st.image("sherwin.jpg", caption="Sherwin Lobo ", width=150)
    st.write("""
    Sherwin Lobo is the Front end Developer at Aegis. He is dedicated to safeguarding your data.
    """)

with col2:
    st.image("sahil.jpeg", caption="Sahil Amundkar", width=150)
    st.write("""
    Sahil Amundkar is an AI Engineer at Aegis. His expertise in developing cutting-edge security technologies keeps our solutions ahead of the curve.
    """)

with col3:
    st.image("deversh.jpeg", caption="Deversh Damani", width=150)
    st.write("""
    Deversh Damani leads our development team, ensuring that our software is robust, secure, and user-friendly.
    """)

st.header("Our Values")
st.write("""
    - **Integrity**: We adhere to the highest ethical standards in all our operations.
    - **Innovation**: We constantly innovate to provide the best cybersecurity solutions.
    - **Excellence**: We strive for excellence in everything we do.
    - **Customer Focus**: Our customers' security and satisfaction are our top priorities.
""")

# Contact Us Section with Icon
st.header("Contact Us")
st.markdown("""
    <style>
        .contact-icon {
            font-size: 24px;
            color: #4CAF50;
            vertical-align: middle;
            margin-right: 10px;
        }
    </style>
    <p>
        <i class="fas fa-envelope contact-icon"></i>
        If you have any questions or would like to learn more about our services, please feel free to contact us:
    </p>
    <ul>
        <li>Email: contact@aegis.com</li>
        <li>Phone: (123) 456-7890</li>
    </ul>
""", unsafe_allow_html=True)

