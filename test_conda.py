# test_conda.py
import streamlit as st
from rdkit import Chem
import pandas as pd

st.set_page_config(page_title="Conda Test", layout="wide")
st.title("✅ Conda Environment Test")

# Test RDKit
mol = Chem.MolFromSmiles("CCO")
st.success(f"RDKit works! Ethanol molecule created: {mol}")

# Test other packages
df = pd.DataFrame({"Test": [1, 2, 3]})
st.dataframe(df)

st.success("🎉 Everything works! Your conda environment is ready.")