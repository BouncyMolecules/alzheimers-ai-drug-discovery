# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.Draw import MolsToGridImage
import py3Dmol
import requests
import sys
import os

# Add the models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import our custom AI model
from smiles_generator import AlzheimerDrugGenerator

# Configure the page
st.set_page_config(
    page_title="AI Alzheimer's Drug Discovery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
        margin: 5px;
    }
    .drug-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🧠 AI-Powered Alzheimer\'s Drug Discovery</h1>', unsafe_allow_html=True)

# Initialize session state variables
if 'generated_molecules' not in st.session_state:
    st.session_state.generated_molecules = []
if 'alz_generator' not in st.session_state:
    st.session_state.alz_generator = AlzheimerDrugGenerator()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Load Alzheimer's drug data
@st.cache_data
def load_alzheimers_data():
    """Load the Alzheimer's drug dataset"""
    try:
        df = pd.read_csv('data/alzheimers_compounds.csv')
        st.success(f"✅ Loaded {len(df)} Alzheimer's compounds")
        return df['smiles'].tolist(), df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        # Fallback data
        fallback_smiles = [
            'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',  # Memantine
            'COC1=CC2=C(C=C1)C=CN2CCOC3=CC=CC=C3',  # Donepezil
            'CN1CCC2=CC=CC=C2C1CC(=O)OC(C)C',  # Rivastigmine
            'CC1=CC(=CC=C1O)CC(C)(C)NCCO',  # Galantamine
        ]
        return fallback_smiles, pd.DataFrame({
            'smiles': fallback_smiles,
            'name': ['Memantine', 'Donepezil', 'Rivastigmine', 'Galantamine'],
            'mechanism': ['NMDA antagonist', 'AChE inhibitor', 'AChE inhibitor', 'AChE inhibitor']
        })

# Sidebar navigation
st.sidebar.title("🔬 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Dashboard",
    "📚 Training Data", 
    "🧠 AI Drug Generator",
    "📊 Molecule Analysis",
    "🔬 3D Visualization",
    "ℹ️ About"
])

# Load data
training_smiles, drug_data = load_alzheimers_data()

# Page 1: Dashboard
if page == "🏠 Dashboard":
    st.header("🚀 Alzheimer's Drug Discovery Pipeline")
    
    st.markdown("""
    Welcome to your AI-powered drug discovery platform! This tool demonstrates how artificial intelligence 
    can learn from existing Alzheimer's drugs to generate novel therapeutic candidates.
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Known Drugs", len(drug_data))
    
    with col2:
        st.metric("AI Model", "Ready" if st.session_state.model_trained else "Not Trained")
    
    with col3:
        generated_count = len(st.session_state.generated_molecules)
        st.metric("Generated", generated_count)
    
    with col4:
        valid_count = len([m for m in st.session_state.generated_molecules 
                          if st.session_state.alz_generator.is_valid_smiles(m)])
        st.metric("Valid Molecules", valid_count)
    
    # Pipeline overview
    st.subheader("🎯 Drug Discovery Pipeline")
    
    pipeline_steps = [
        ("1. Data Collection", "Gather known Alzheimer's drugs"),
        ("2. AI Training", "Train model on drug structures"), 
        ("3. Molecule Generation", "AI creates novel candidates"),
        ("4. Property Analysis", "Evaluate drug-like properties"),
        ("5. Lead Selection", "Identify promising candidates")
    ]
    
    for step, description in pipeline_steps:
        with st.expander(f"📋 {step}"):
            st.write(description)
    
    # Quick start
    st.subheader("🚀 Quick Start")
    st.info("""
    **To get started:**
    1. Go to **'Training Data'** to see the drugs the AI learns from
    2. Visit **'AI Drug Generator'** to train the model and create new molecules
    3. Use **'Molecule Analysis'** to evaluate the generated compounds
    """)

# Page 2: Training Data
elif page == "📚 Training Data":
    st.header("📚 Alzheimer's Drug Training Data")
    
    st.markdown("""
    This is the data the AI model learns from - known Alzheimer's drugs and related compounds 
    with their mechanisms of action.
    """)
    
    # Display the dataset
    st.subheader("🧪 Compound Library")
    st.dataframe(drug_data, use_container_width=True)
    
    # Show molecular structures
    st.subheader("🧬 Molecular Structures")
    
    # Convert SMILES to molecules
    molecules = []
    valid_molecules = []
    
    for smiles in drug_data['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append(mol)
            valid_molecules.append(smiles)
    
    if molecules:
        # Display in a grid
        img = Draw.MolsToGridImage(
            molecules,
            molsPerRow=4,
            subImgSize=(250, 250),
            legends=list(drug_data['name']),
            returnPNG=False
        )
        st.image(img, use_column_width=True)
    
    # Data statistics
    st.subheader("📊 Data Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mechanism Distribution**")
        mechanism_counts = drug_data['mechanism'].value_counts()
        st.bar_chart(mechanism_counts)
    
    with col2:
        st.write("**Property Ranges**")
        properties_data = []
        for smiles in valid_molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                props = {
                    'MW': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                }
                properties_data.append(props)
        
        if properties_data:
            props_df = pd.DataFrame(properties_data)
            st.write(f"Molecular Weight: {props_df['MW'].min():.1f} - {props_df['MW'].max():.1f}")
            st.write(f"LogP: {props_df['LogP'].min():.1f} - {props_df['LogP'].max():.1f}")

# Page 3: AI Drug Generator
elif page == "🧠 AI Drug Generator":
    st.header("🧠 AI-Powered Drug Generation")
    
    st.markdown("""
    This is where the magic happens! The AI model learns the chemical patterns from known Alzheimer's drugs 
    and generates novel compounds that could become new therapeutics.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Model Training")
        
        st.info("""
        **Training Process:**
        - The AI learns character-by-character how to build valid drug molecules
        - It studies patterns from the known Alzheimer's drugs
        - After training, it can generate entirely new molecules
        """)
        
        training_col1, training_col2 = st.columns(2)
        
        with training_col1:
            epochs = st.slider("Training Epochs", 100, 1000, 300, 50)
        
        with training_col2:
            learning_rate = st.selectbox("Learning Rate", [0.1, 0.01, 0.001, 0.0001], index=2)
        
        if st.button("🎯 Train AI Model", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is learning from Alzheimer's drug structures..."):
                try:
                    st.session_state.alz_generator.train(
                        training_smiles, 
                        epochs=epochs, 
                        lr=learning_rate
                    )
                    st.session_state.model_trained = True
                    st.success("✅ Model trained successfully! The AI has learned Alzheimer's drug patterns.")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
    
    with col2:
        st.subheader("🎯 Generation Settings")
        
        if st.session_state.model_trained:
            num_molecules = st.slider("Molecules to Generate", 1, 20, 10)
            temperature = st.slider("Creativity", 0.1, 1.5, 0.8, 0.1)
            
            st.info(f"""
            **Temperature: {temperature}**
            - Low ({temperature < 0.5}): Conservative, similar to training data
            - Medium (0.5-1.0): Balanced creativity  
            - High (>1.0): More novel, potentially unusual
            """)
            
            if st.button("🧪 Generate New Drugs", use_container_width=True):
                with st.spinner("🔄 AI is designing novel Alzheimer's drug candidates..."):
                    try:
                        new_molecules = st.session_state.alz_generator.generate_molecules(
                            num_molecules, 
                            temperature
                        )
                        st.session_state.generated_molecules = new_molecules
                        
                        if new_molecules:
                            st.success(f"✅ Generated {len(new_molecules)} novel compounds!")
                        else:
                            st.warning("⚠️ No valid molecules generated. Try increasing temperature.")
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
        else:
            st.warning("⚠️ Please train the model first!")
    
    # Display generated molecules
    if st.session_state.generated_molecules:
        st.subheader("🎉 AI-Generated Drug Candidates")
        
        # Calculate properties
        results = []
        valid_molecules = []
        
        for i, smiles in enumerate(st.session_state.generated_molecules):
            props = st.session_state.alz_generator.calculate_drug_properties(smiles)
            if props:
                props['smiles'] = smiles
                props['molecule_id'] = f"ALZ-{i+1:03d}"
                props['drug_likeness'] = "✅ PASS" if props['passes_lipinski'] else "❌ FAIL"
                results.append(props)
                valid_molecules.append(Chem.MolFromSmiles(smiles))
        
        if results:
            # Display results table
            results_df = pd.DataFrame(results)
            display_columns = ['molecule_id', 'smiles', 'molecular_weight', 'logp', 
                             'hydrogen_bond_donors', 'hydrogen_bond_acceptors', 'drug_likeness']
            
            st.dataframe(results_df[display_columns], use_container_width=True)
            
            # Show molecular structures
            st.subheader("🧬 Generated Molecular Structures")
            if valid_molecules:
                img = Draw.MolsToGridImage(
                    valid_molecules[:12],
                    molsPerRow=4,
                    subImgSize=(250, 250),
                    legends=[f"ALZ-{i+1:03d}" for i in range(len(valid_molecules[:12]))],
                    returnPNG=False
                )
                st.image(img, use_container_width=True)

# Page 4: Molecule Analysis
elif page == "📊 Molecule Analysis":
    st.header("📊 Molecular Property Analysis")
    
    if not st.session_state.generated_molecules:
        st.warning("⚠️ No generated molecules to analyze. Please generate some molecules first.")
    else:
        st.success(f"📈 Analyzing {len(st.session_state.generated_molecules)} generated molecules")
        
        # Calculate properties
        results = []
        for i, smiles in enumerate(st.session_state.generated_molecules):
            props = st.session_state.alz_generator.calculate_drug_properties(smiles)
            if props:
                props['molecule_id'] = f"ALZ-{i+1:03d}"
                props['smiles'] = smiles
                results.append(props)
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Summary metrics
            st.subheader("📋 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_mw = results_df['molecular_weight'].mean()
                st.metric("Avg Molecular Weight", f"{avg_mw:.1f}")
            
            with col2:
                avg_logp = results_df['logp'].mean()
                st.metric("Avg LogP", f"{avg_logp:.2f}")
            
            with col3:
                passes_lipinski = results_df['passes_lipinski'].sum()
                st.metric("Pass Lipinski", f"{passes_lipinski}/{len(results_df)}")
            
            with col4:
                avg_tpsa = results_df['polar_surface_area'].mean()
                st.metric("Avg TPSA", f"{avg_tpsa:.1f}")
            
            # Property distributions
            st.subheader("📊 Property Distributions")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Molecular weight distribution
            ax1.hist(results_df['molecular_weight'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=500, color='red', linestyle='--', label='Lipinski Limit (500)')
            ax1.set_xlabel('Molecular Weight')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Molecular Weight Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # LogP distribution
            ax2.hist(results_df['logp'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.axvline(x=5, color='red', linestyle='--', label='Lipinski Limit (5)')
            ax2.set_xlabel('LogP')
            ax2.set_ylabel('Frequency')
            ax2.set_title('LogP Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Hydrogen bond donors vs acceptors
            ax3.scatter(results_df['hydrogen_bond_donors'], results_df['hydrogen_bond_acceptors'], 
                       alpha=0.6, color='coral', s=60)
            ax3.set_xlabel('H-Bond Donors')
            ax3.set_ylabel('H-Bond Acceptors')
            ax3.set_title('H-Bond Donors vs Acceptors')
            ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='HBA Limit')
            ax3.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='HBD Limit')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Property correlations
            correlation = results_df[['molecular_weight', 'logp', 'polar_surface_area']].corr()
            im = ax4.imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(correlation.columns)))
            ax4.set_yticks(range(len(correlation.columns)))
            ax4.set_xticklabels(correlation.columns, rotation=45)
            ax4.set_yticklabels(correlation.columns)
            ax4.set_title('Property Correlations')
            
            # Add correlation values to heatmap
            for i in range(len(correlation.columns)):
                for j in range(len(correlation.columns)):
                    ax4.text(j, i, f'{correlation.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='black', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed table
            st.subheader("📋 Detailed Properties")
            st.dataframe(results_df, use_container_width=True)

# Page 5: 3D Visualization
elif page == "🔬 3D Visualization":
    st.header("🔬 3D Molecular Visualization")
    
    if not st.session_state.generated_molecules:
        st.warning("⚠️ No generated molecules to visualize. Please generate some molecules first.")
    else:
        # Molecule selector
        selected_molecule = st.selectbox(
            "Select Molecule to Visualize",
            options=range(len(st.session_state.generated_molecules)),
            format_func=lambda x: f"ALZ-{x+1:03d}: {st.session_state.generated_molecules[x]}"
        )
        
        if selected_molecule is not None:
            smiles = st.session_state.generated_molecules[selected_molecule]
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                # Generate 3D coordinates
                mol_3d = Chem.AddHs(mol)  # Add hydrogens
                AllChem.EmbedMolecule(mol_3d)  # Generate 3D coordinates
                AllChem.MMFFOptimizeMolecule(mol_3d)  # Energy minimization
                
                # Convert to PDB format for 3D visualization
                pdb_block = Chem.MolToPDBBlock(mol_3d)
                
                # Create 3D viewer
                st.subheader(f"3D Structure: ALZ-{selected_molecule+1:03d}")
                
                viewer = py3Dmol.view(width=600, height=400)
                viewer.addModel(pdb_block, 'pdb')
                viewer.setStyle({'stick': {}})
                viewer.setBackgroundColor('white')
                viewer.zoomTo()
                
                # Display the 3D viewer
                st.components.v1.html(viewer._make_html(), height=500)
                
                # Molecular properties
                props = st.session_state.alz_generator.calculate_drug_properties(smiles)
                if props:
                    st.subheader("Molecular Properties")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Molecular Weight:** {props['molecular_weight']:.2f}")
                        st.write(f"**LogP:** {props['logp']:.2f}")
                        st.write(f"**H-Bond Donors:** {props['hydrogen_bond_donors']}")
                    
                    with col2:
                        st.write(f"**H-Bond Acceptors:** {props['hydrogen_bond_acceptors']}")
                        st.write(f"**Rotatable Bonds:** {props['rotatable_bonds']}")
                        st.write(f"**TPSA:** {props['polar_surface_area']:.2f}")
                    
                    # Drug-likeness assessment
                    if props['passes_lipinski']:
                        st.success("✅ This molecule passes Lipinski's Rule of Five")
                    else:
                        st.warning("⚠️ This molecule fails Lipinski's Rule of Five")

# Page 6: About
else:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🧠 AI-Powered Alzheimer's Drug Discovery
    
    This platform demonstrates how artificial intelligence can accelerate drug discovery for Alzheimer's disease.
    
    ### 🔬 How It Works
    
    1. **Data Learning**: The AI studies known Alzheimer's drugs and their chemical structures
    2. **Pattern Recognition**: It learns the "chemical grammar" of effective drugs
    3. **Novel Generation**: The AI creates new molecular structures that follow these patterns
    4. **Property Prediction**: Each generated molecule is evaluated for drug-like properties
    
    ### 🎯 Scientific Basis
    
    - **Training Data**: Known Alzheimer's drugs (Memantine, Donepezil, etc.) and related compounds
    - **AI Model**: LSTM neural network that learns SMILES string patterns
    - **Evaluation**: Lipinski's Rule of Five and other drug-like property filters
    
    ### 🚀 Potential Impact
    
    This approach could:
    - Discover novel therapeutic candidates faster
    - Explore chemical space more efficiently
    - Reduce early-stage drug discovery costs
    - Identify promising leads for laboratory testing
    
    ### 🛠️ Technology Stack
    
    - **Streamlit**: Web application framework
    - **PyTorch**: Deep learning framework
    - **RDKit**: Cheminformatics toolkit
    - **py3Dmol**: 3D molecular visualization
    
    ---
    
    *Built for educational and research purposes to demonstrate AI in drug discovery.*
    """)

# Footer
st.markdown("---")
st.markdown(
    "🧠 *AI Alzheimer's Drug Discovery Platform v1.0 | "
    "Built with Streamlit, PyTorch & RDKit*"
)