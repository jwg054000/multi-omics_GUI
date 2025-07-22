#!/usr/bin/env python3
"""
🧬 Prairie Genomics Suite - Streamlit Cloud Ready
Advanced Genomics Analysis Platform

A comprehensive, publication-ready genomics analysis platform optimized for Streamlit Cloud deployment.
Features include differential expression analysis, survival analysis, pathway enrichment,
literature search, and publication-quality visualizations.

Usage:
    streamlit run prairie_genomics_streamlit_ready.py

Author: Prairie Genomics Team
Version: 2.1.0 - Streamlit Cloud Ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import sys
import os
import io
from datetime import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Prairie Genomics Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/prairie-genomics/suite',
        'Report a bug': "https://github.com/prairie-genomics/suite/issues",
        'About': "Prairie Genomics Suite - Making genomics analysis accessible to every researcher"
    }
)

# Handle Streamlit version compatibility
def safe_rerun():
    """Safe rerun function that works with different Streamlit versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.info("Please refresh the page to see the changes")

# ================================
# CORE FUNCTIONALITY - BUILT-IN
# ================================

class BasicStatsAnalyzer:
    """Built-in statistical analysis for when external packages aren't available"""
    
    @staticmethod
    def ttest_analysis(data: pd.DataFrame, group1_samples: List[str], group2_samples: List[str]):
        """Perform basic t-test analysis"""
        from scipy import stats
        
        results = []
        for gene in data.index:
            try:
                group1_values = data.loc[gene, group1_samples].dropna()
                group2_values = data.loc[gene, group2_samples].dropna()
                
                if len(group1_values) < 3 or len(group2_values) < 3:
                    continue
                    
                # T-test
                t_stat, p_value = stats.ttest_ind(group1_values, group2_values)
                
                # Log fold change
                mean1 = np.mean(group1_values)
                mean2 = np.mean(group2_values)
                log_fold_change = np.log2((mean2 + 1) / (mean1 + 1))
                
                # Base mean
                base_mean = (mean1 + mean2) / 2
                
                results.append({
                    'Gene': gene,
                    'baseMean': base_mean,
                    'log2FoldChange': log_fold_change,
                    'pvalue': p_value,
                    'padj': p_value,  # No multiple testing correction for simplicity
                    'mean_group1': mean1,
                    'mean_group2': mean2
                })
                
            except Exception as e:
                continue
                
        return pd.DataFrame(results)

class BasicSurvivalAnalyzer:
    """Built-in survival analysis"""
    
    @staticmethod
    def kaplan_meier_analysis(clinical_data: pd.DataFrame, time_col: str, event_col: str, group_col: str = None):
        """Basic Kaplan-Meier survival analysis"""
        try:
            from lifelines import KaplanMeierFitter
            import matplotlib.pyplot as plt
            
            kmf = KaplanMeierFitter()
            
            if group_col and group_col in clinical_data.columns:
                # Group-based analysis
                fig, ax = plt.subplots(figsize=(10, 6))
                
                groups = clinical_data[group_col].unique()
                for group in groups:
                    mask = clinical_data[group_col] == group
                    group_data = clinical_data[mask]
                    
                    if len(group_data) > 5:  # Minimum sample size
                        kmf.fit(
                            durations=group_data[time_col],
                            event_observed=group_data[event_col],
                            label=f'{group} (n={len(group_data)})'
                        )
                        kmf.plot_survival_function(ax=ax)
                
                ax.set_title('Kaplan-Meier Survival Curves')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Survival Probability')
                ax.grid(True, alpha=0.3)
                
                return fig
            else:
                # Overall survival
                fig, ax = plt.subplots(figsize=(10, 6))
                
                kmf.fit(
                    durations=clinical_data[time_col],
                    event_observed=clinical_data[event_col]
                )
                kmf.plot_survival_function(ax=ax)
                
                ax.set_title('Overall Survival Curve')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Survival Probability')
                ax.grid(True, alpha=0.3)
                
                return fig
                
        except ImportError:
            st.warning("Lifelines not available. Please install: pip install lifelines")
            return None
        except Exception as e:
            st.error(f"Survival analysis failed: {e}")
            return None

class BasicPathwayAnalyzer:
    """Built-in pathway analysis with basic gene sets"""
    
    @staticmethod
    def get_basic_pathways():
        """Return basic pathway gene sets"""
        return {
            "Cell Cycle": ["CDK1", "CDK2", "CDK4", "CDK6", "CDKN1A", "CDKN1B", "CDKN2A", "CCND1", "CCNE1", "RB1", "E2F1"],
            "Apoptosis": ["TP53", "BAX", "BCL2", "CASP3", "CASP8", "CASP9", "PARP1", "APAF1", "CYCS", "BAK1"],
            "DNA Repair": ["BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51", "XRCC1", "ERCC1", "MLH1"],
            "PI3K/AKT Pathway": ["PIK3CA", "PIK3CB", "AKT1", "AKT2", "PTEN", "MTOR", "GSK3B", "FOXO1", "PDK1"],
            "MAPK Pathway": ["KRAS", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "RAF1", "EGFR", "GRB2"],
            "Immune Response": ["CD3D", "CD4", "CD8A", "IFNG", "IL2", "TNF", "CTLA4", "PDCD1", "CD274", "LAG3"],
            "Metabolism": ["GLUT1", "HK2", "PKM", "LDHA", "PDK1", "ACLY", "FASN", "CPT1A", "PPARA", "SREBF1"]
        }
    
    @staticmethod
    def enrichment_analysis(gene_list: List[str], background_size: int = 20000):
        """Basic pathway enrichment analysis"""
        from scipy import stats
        
        pathways = BasicPathwayAnalyzer.get_basic_pathways()
        results = []
        
        for pathway_name, pathway_genes in pathways.items():
            # Calculate overlap
            overlap = set(gene_list) & set(pathway_genes)
            overlap_count = len(overlap)
            
            if overlap_count == 0:
                continue
                
            # Hypergeometric test
            total_genes = len(gene_list)
            pathway_size = len(pathway_genes)
            
            # P-value calculation (hypergeometric)
            p_value = stats.hypergeom.sf(overlap_count - 1, background_size, pathway_size, total_genes)
            
            # Enrichment score
            expected = (total_genes * pathway_size) / background_size
            enrichment_score = overlap_count / expected if expected > 0 else 0
            
            results.append({
                'Pathway': pathway_name,
                'Overlap': overlap_count,
                'Pathway_Size': pathway_size,
                'P_value': p_value,
                'Enrichment_Score': enrichment_score,
                'Genes': ', '.join(sorted(overlap))
            })
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('P_value')
            df['FDR'] = df['P_value'] * len(df)  # Bonferroni correction
        
        return df

# ================================
# LAZY LOADING FUNCTIONS
# ================================

@st.cache_resource
def get_plotting_libs():
    """Lazy load plotting libraries"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import seaborn as sns
        import matplotlib.pyplot as plt
        return px, go, make_subplots, sns, plt
    except ImportError as e:
        st.error(f"Plotting libraries not available: {e}")
        return None, None, None, None, None

@st.cache_resource
def get_optional_libraries():
    """Load optional libraries with graceful fallback"""
    libs = {}
    
    # Gene conversion
    try:
        import mygene
        libs['mygene'] = mygene
    except ImportError:
        pass
    
    # Advanced stats
    try:
        from scipy import stats
        libs['scipy_stats'] = stats
    except ImportError:
        pass
    
    # Machine learning
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        libs['pca'] = PCA
        libs['scaler'] = StandardScaler
    except ImportError:
        pass
    
    # Pathway analysis
    try:
        import gseapy as gp
        libs['gseapy'] = gp
    except ImportError:
        pass
    
    return libs

# ================================
# MAIN APPLICATION CLASS
# ================================

class PrairieGenomicsStreamlit:
    """Main Streamlit application class optimized for cloud deployment"""
    
    def __init__(self):
        """Initialize the application"""
        self.setup_session_state()
        self.libs = get_optional_libraries()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'expression_data' not in st.session_state:
            st.session_state.expression_data = None
        if 'clinical_data' not in st.session_state:
            st.session_state.clinical_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'p_threshold': 0.05,
                'fc_threshold': 1.5,
                'min_expression': 1.0
            }
    
    def show_header(self):
        """Display application header and navigation"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1f4037, #99f2c8);'>
            <h1 style='color: white; margin: 0; font-size: 3rem;'>🧬 Prairie Genomics Suite</h1>
            <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Streamlit Cloud Ready Edition</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for navigation
        return st.tabs([
            "📊 Data Import", 
            "🔍 Gene Analysis", 
            "📈 Differential Expression", 
            "⏱️ Survival Analysis",
            "🛤️ Pathway Analysis", 
            "📚 Literature Search",
            "🎨 Visualizations", 
            "💾 Export Results",
            "⚙️ Settings"
        ])
    
    def show_sidebar(self):
        """Display sidebar with quick info and controls"""
        with st.sidebar:
            st.markdown("### 🎛️ Quick Controls")
            
            # Data status
            st.markdown("#### Data Status")
            expr_status = "✅ Loaded" if st.session_state.expression_data is not None else "❌ Not loaded"
            clin_status = "✅ Loaded" if st.session_state.clinical_data is not None else "❌ Not loaded"
            
            st.markdown(f"""
            - **Expression Data:** {expr_status}
            - **Clinical Data:** {clin_status}
            """)
            
            # Quick actions
            st.markdown("#### Quick Actions")
            if st.button("🗑️ Clear All Data"):
                for key in ['expression_data', 'clinical_data', 'analysis_results']:
                    st.session_state[key] = None if key != 'analysis_results' else {}
                safe_rerun()
            
            # Load example data
            if st.button("📋 Load Example Data"):
                self.load_example_data()
            
            # Settings preview
            st.markdown("#### Current Settings")
            st.json(st.session_state.settings)
            
            # About section
            st.markdown("---")
            st.markdown("""
            ### 🧬 About Prairie Genomics
            
            **Version:** 2.1.0 (Streamlit Cloud Ready)
            
            **Features:**
            - Differential expression analysis
            - Survival analysis
            - Pathway enrichment
            - Literature search
            - Publication-quality plots
            - Data export capabilities
            
            **Optimized for:** Streamlit Cloud deployment with graceful fallbacks for missing dependencies.
            """)
    
    def load_example_data(self):
        """Load example genomic data for demonstration"""
        try:
            # Generate synthetic expression data
            np.random.seed(42)
            genes = [f"GENE_{i:04d}" for i in range(1000)]
            samples = [f"Sample_{i:02d}" for i in range(50)]
            
            # Create expression matrix with some differential patterns
            expression_data = pd.DataFrame(
                np.random.lognormal(mean=5, sigma=1, size=(len(genes), len(samples))),
                index=genes,
                columns=samples
            )
            
            # Make some genes differentially expressed
            diff_genes = genes[:100]
            treatment_samples = samples[25:]
            for gene in diff_genes:
                if np.random.random() > 0.5:
                    expression_data.loc[gene, treatment_samples] *= np.random.uniform(1.5, 3.0)
                else:
                    expression_data.loc[gene, treatment_samples] *= np.random.uniform(0.3, 0.7)
            
            # Create clinical data
            clinical_data = pd.DataFrame({
                'Sample_ID': samples,
                'Group': ['Control'] * 25 + ['Treatment'] * 25,
                'Age': np.random.randint(30, 80, len(samples)),
                'Sex': np.random.choice(['M', 'F'], len(samples)),
                'Overall_Survival_Days': np.random.randint(100, 2000, len(samples)),
                'Event': np.random.choice([0, 1], len(samples), p=[0.6, 0.4])
            })
            clinical_data.set_index('Sample_ID', inplace=True)
            
            # Store in session state
            st.session_state.expression_data = expression_data
            st.session_state.clinical_data = clinical_data
            
            st.success("✅ Example data loaded successfully!")
            st.info(f"Loaded {len(genes)} genes across {len(samples)} samples")
            
        except Exception as e:
            st.error(f"Failed to load example data: {e}")
    
    # ================================
    # TAB SECTIONS
    # ================================
    
    def data_import_section(self, tab):
        """Data import and loading section"""
        with tab:
            st.header("📊 Data Import")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Expression Data")
                expr_file = st.file_uploader(
                    "Upload expression matrix (CSV/Excel)",
                    type=['csv', 'xlsx', 'txt'],
                    key="expr_upload"
                )
                
                if expr_file:
                    try:
                        if expr_file.name.endswith('.xlsx'):
                            data = pd.read_excel(expr_file, index_col=0)
                        else:
                            data = pd.read_csv(expr_file, index_col=0, sep=None, engine='python')
                        
                        st.session_state.expression_data = data
                        st.success(f"✅ Loaded {data.shape[0]} genes, {data.shape[1]} samples")
                        st.dataframe(data.head())
                        
                    except Exception as e:
                        st.error(f"Error loading expression data: {e}")
            
            with col2:
                st.subheader("Clinical Data")
                clin_file = st.file_uploader(
                    "Upload clinical metadata (CSV/Excel)",
                    type=['csv', 'xlsx', 'txt'],
                    key="clin_upload"
                )
                
                if clin_file:
                    try:
                        if clin_file.name.endswith('.xlsx'):
                            data = pd.read_excel(clin_file, index_col=0)
                        else:
                            data = pd.read_csv(clin_file, index_col=0, sep=None, engine='python')
                        
                        st.session_state.clinical_data = data
                        st.success(f"✅ Loaded {data.shape[0]} samples, {data.shape[1]} variables")
                        st.dataframe(data.head())
                        
                    except Exception as e:
                        st.error(f"Error loading clinical data: {e}")
            
            # Data preview section
            if st.session_state.expression_data is not None or st.session_state.clinical_data is not None:
                st.markdown("---")
                st.subheader("📋 Data Preview")
                
                if st.session_state.expression_data is not None:
                    with st.expander("Expression Data Preview"):
                        st.write(f"Shape: {st.session_state.expression_data.shape}")
                        st.dataframe(st.session_state.expression_data.head(10))
                
                if st.session_state.clinical_data is not None:
                    with st.expander("Clinical Data Preview"):
                        st.write(f"Shape: {st.session_state.clinical_data.shape}")
                        st.dataframe(st.session_state.clinical_data.head(10))
    
    def gene_conversion_section(self, tab):
        """Gene ID conversion section"""
        with tab:
            st.header("🔍 Gene Analysis & Conversion")
            
            if 'mygene' in self.libs:
                st.info("✅ Gene conversion service available (mygene)")
                
                if st.session_state.expression_data is not None:
                    genes = list(st.session_state.expression_data.index[:100])  # Sample first 100
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        conversion_type = st.selectbox(
                            "Conversion Type",
                            ["symbol", "entrezgene", "ensembl.gene"]
                        )
                        
                        if st.button("🔄 Convert Gene IDs"):
                            try:
                                mg = self.libs['mygene']
                                results = mg.querymany(genes, scopes='symbol', fields=conversion_type, species='human')
                                
                                conversion_df = pd.DataFrame(results)
                                st.session_state.analysis_results['gene_conversion'] = conversion_df
                                st.success("Gene conversion completed!")
                                
                            except Exception as e:
                                st.error(f"Gene conversion failed: {e}")
                    
                    with col2:
                        if 'gene_conversion' in st.session_state.analysis_results:
                            st.dataframe(st.session_state.analysis_results['gene_conversion'])
            else:
                st.warning("Gene conversion service not available. Install mygene: pip install mygene")
                
                # Show basic gene analysis instead
                if st.session_state.expression_data is not None:
                    st.subheader("📊 Gene Expression Summary")
                    
                    expr_data = st.session_state.expression_data
                    
                    # Basic statistics
                    st.write("**Expression Statistics:**")
                    stats_df = pd.DataFrame({
                        'Mean': expr_data.mean(axis=1),
                        'Std': expr_data.std(axis=1),
                        'Min': expr_data.min(axis=1),
                        'Max': expr_data.max(axis=1),
                        'CV': expr_data.std(axis=1) / expr_data.mean(axis=1)
                    }).round(3)
                    
                    st.dataframe(stats_df.head(20))
                    
                    # Gene filtering options
                    st.subheader("🔧 Gene Filtering")
                    min_expr = st.slider("Minimum average expression", 0.0, 10.0, 1.0, 0.1)
                    max_cv = st.slider("Maximum coefficient of variation", 0.0, 5.0, 2.0, 0.1)
                    
                    if st.button("Apply Filters"):
                        mean_expr = expr_data.mean(axis=1)
                        cv = expr_data.std(axis=1) / mean_expr
                        
                        filtered_genes = expr_data.index[
                            (mean_expr >= min_expr) & (cv <= max_cv)
                        ]
                        
                        filtered_data = expr_data.loc[filtered_genes]
                        st.session_state.expression_data = filtered_data
                        
                        st.success(f"Filtered to {len(filtered_genes)} genes (from {len(expr_data.index)})")
                        safe_rerun()
    
    def differential_expression_section(self, tab):
        """Differential expression analysis section"""
        with tab:
            st.header("📈 Differential Expression Analysis")
            
            if st.session_state.expression_data is None:
                st.warning("Please load expression data first!")
                return
            
            if st.session_state.clinical_data is None:
                st.warning("Please load clinical data first!")
                return
            
            # Group selection
            st.subheader("👥 Group Definition")
            
            clinical_cols = list(st.session_state.clinical_data.columns)
            group_column = st.selectbox("Select grouping variable:", clinical_cols)
            
            if group_column:
                unique_groups = st.session_state.clinical_data[group_column].unique()
                
                col1, col2 = st.columns(2)
                with col1:
                    control_group = st.selectbox("Control group:", unique_groups)
                with col2:
                    treatment_groups = unique_groups[unique_groups != control_group]
                    treatment_group = st.selectbox("Treatment group:", treatment_groups)
                
                # Analysis parameters
                st.subheader("🔧 Analysis Parameters")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    p_threshold = st.number_input("P-value threshold:", 0.001, 0.1, 0.05, 0.001)
                with col2:
                    fc_threshold = st.number_input("Fold change threshold:", 1.1, 5.0, 1.5, 0.1)
                with col3:
                    min_expr = st.number_input("Minimum expression:", 0.0, 10.0, 1.0, 0.1)
                
                # Run analysis button
                if st.button("🚀 Run Differential Expression Analysis"):
                    with st.spinner("Running analysis..."):
                        try:
                            # Get sample lists
                            control_samples = st.session_state.clinical_data[
                                st.session_state.clinical_data[group_column] == control_group
                            ].index.tolist()
                            
                            treatment_samples = st.session_state.clinical_data[
                                st.session_state.clinical_data[group_column] == treatment_group
                            ].index.tolist()
                            
                            # Filter for common samples
                            expr_samples = set(st.session_state.expression_data.columns)
                            control_samples = [s for s in control_samples if s in expr_samples]
                            treatment_samples = [s for s in treatment_samples if s in expr_samples]
                            
                            if len(control_samples) < 3 or len(treatment_samples) < 3:
                                st.error("Need at least 3 samples per group!")
                                return
                            
                            # Run analysis
                            analyzer = BasicStatsAnalyzer()
                            results = analyzer.ttest_analysis(
                                st.session_state.expression_data,
                                control_samples,
                                treatment_samples
                            )
                            
                            if not results.empty:
                                # Apply filters
                                significant = (
                                    (results['padj'] < p_threshold) &
                                    (np.abs(results['log2FoldChange']) > np.log2(fc_threshold)) &
                                    (results['baseMean'] > min_expr)
                                )
                                
                                results['Significant'] = significant
                                results['Regulation'] = results['log2FoldChange'].apply(
                                    lambda x: 'Up' if x > 0 else 'Down'
                                )
                                
                                # Store results
                                st.session_state.analysis_results['differential_expression'] = results
                                
                                # Display summary
                                n_total = len(results)
                                n_significant = significant.sum()
                                n_up = ((results['log2FoldChange'] > 0) & significant).sum()
                                n_down = ((results['log2FoldChange'] < 0) & significant).sum()
                                
                                st.success("✅ Analysis completed!")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total Genes", n_total)
                                col2.metric("Significant", n_significant)
                                col3.metric("Upregulated", n_up)
                                col4.metric("Downregulated", n_down)
                                
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
            
            # Display results
            if 'differential_expression' in st.session_state.analysis_results:
                st.markdown("---")
                st.subheader("📊 Results")
                
                results = st.session_state.analysis_results['differential_expression']
                
                # Results table
                with st.expander("📋 Detailed Results Table"):
                    st.dataframe(
                        results.sort_values('padj').head(100),
                        use_container_width=True
                    )
                
                # Volcano plot
                if len(results) > 0:
                    self.create_volcano_plot(results)
    
    def create_volcano_plot(self, results):
        """Create volcano plot"""
        try:
            px, go, make_subplots, sns, plt = get_plotting_libs()
            if px is None:
                st.warning("Plotting libraries not available")
                return
            
            # Prepare data
            results_plot = results.copy()
            results_plot['-log10(pvalue)'] = -np.log10(results_plot['pvalue'] + 1e-300)
            
            # Color by significance
            colors = []
            for _, row in results_plot.iterrows():
                if row['Significant']:
                    colors.append('Up-regulated' if row['log2FoldChange'] > 0 else 'Down-regulated')
                else:
                    colors.append('Not significant')
            
            results_plot['Color'] = colors
            
            # Create plot
            fig = px.scatter(
                results_plot,
                x='log2FoldChange',
                y='-log10(pvalue)',
                color='Color',
                hover_name='Gene',
                hover_data=['baseMean', 'pvalue'],
                title='Volcano Plot - Differential Expression',
                color_discrete_map={
                    'Up-regulated': '#FF6B6B',
                    'Down-regulated': '#4ECDC4', 
                    'Not significant': '#95A5A6'
                }
            )
            
            fig.update_layout(
                xaxis_title='log2 Fold Change',
                yaxis_title='-log10(p-value)',
                height=600
            )
            
            # Add threshold lines
            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
            fig.add_vline(x=np.log2(1.5), line_dash="dash", line_color="gray")
            fig.add_vline(x=-np.log2(1.5), line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create volcano plot: {e}")
    
    def survival_analysis_section(self, tab):
        """Survival analysis section"""
        with tab:
            st.header("⏱️ Survival Analysis")
            
            if st.session_state.clinical_data is None:
                st.warning("Please load clinical data first!")
                return
            
            clinical_data = st.session_state.clinical_data
            
            # Column selection
            st.subheader("📊 Configure Survival Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                time_columns = [col for col in clinical_data.columns if 
                              any(keyword in col.lower() for keyword in ['time', 'day', 'month', 'survival'])]
                time_col = st.selectbox("Time to event column:", clinical_data.columns, 
                                      index=clinical_data.columns.get_loc(time_columns[0]) if time_columns else 0)
            
            with col2:
                event_columns = [col for col in clinical_data.columns if 
                               any(keyword in col.lower() for keyword in ['event', 'death', 'status'])]
                event_col = st.selectbox("Event column (1=event, 0=censored):", clinical_data.columns,
                                       index=clinical_data.columns.get_loc(event_columns[0]) if event_columns else 0)
            
            with col3:
                group_col = st.selectbox("Grouping variable (optional):", 
                                       ["None"] + list(clinical_data.columns))
                if group_col == "None":
                    group_col = None
            
            # Run analysis
            if st.button("📈 Run Survival Analysis"):
                try:
                    analyzer = BasicSurvivalAnalyzer()
                    fig = analyzer.kaplan_meier_analysis(clinical_data, time_col, event_col, group_col)
                    
                    if fig:
                        st.pyplot(fig)
                        st.session_state.analysis_results['survival_plot'] = fig
                    else:
                        st.error("Survival analysis failed. Check your data and column selections.")
                        
                except Exception as e:
                    st.error(f"Survival analysis failed: {e}")
            
            # Data preview
            if time_col and event_col:
                st.subheader("📋 Data Preview")
                preview_data = clinical_data[[time_col, event_col]]
                if group_col:
                    preview_data[group_col] = clinical_data[group_col]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Data Summary:")
                    st.write(preview_data.describe())
                
                with col2:
                    st.write("Sample Data:")
                    st.dataframe(preview_data.head(10))
    
    def pathway_analysis_section(self, tab):
        """Pathway analysis section"""
        with tab:
            st.header("🛤️ Pathway Analysis")
            
            if 'differential_expression' not in st.session_state.analysis_results:
                st.warning("Please run differential expression analysis first!")
                return
            
            de_results = st.session_state.analysis_results['differential_expression']
            
            # Gene list options
            st.subheader("🎯 Gene List Selection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                list_type = st.selectbox(
                    "Select gene list:",
                    ["All significant", "Upregulated only", "Downregulated only", "Top 100 by p-value"]
                )
            
            with col2:
                p_cutoff = st.number_input("P-value cutoff:", 0.001, 0.1, 0.05, 0.001)
            
            # Generate gene list based on selection
            if list_type == "All significant":
                gene_list = de_results[de_results['Significant']]['Gene'].tolist()
            elif list_type == "Upregulated only":
                gene_list = de_results[
                    (de_results['Significant']) & (de_results['log2FoldChange'] > 0)
                ]['Gene'].tolist()
            elif list_type == "Downregulated only":
                gene_list = de_results[
                    (de_results['Significant']) & (de_results['log2FoldChange'] < 0)
                ]['Gene'].tolist()
            else:  # Top 100
                gene_list = de_results.nsmallest(100, 'pvalue')['Gene'].tolist()
            
            st.info(f"Selected {len(gene_list)} genes for pathway analysis")
            
            # Run pathway analysis
            if st.button("🚀 Run Pathway Analysis") and len(gene_list) > 0:
                with st.spinner("Running pathway enrichment..."):
                    try:
                        analyzer = BasicPathwayAnalyzer()
                        pathway_results = analyzer.enrichment_analysis(gene_list)
                        
                        if not pathway_results.empty:
                            st.session_state.analysis_results['pathway_analysis'] = pathway_results
                            st.success("✅ Pathway analysis completed!")
                            
                            # Display results
                            st.subheader("📊 Enriched Pathways")
                            
                            # Filter by significance
                            significant_pathways = pathway_results[pathway_results['P_value'] < 0.05]
                            
                            if not significant_pathways.empty:
                                st.dataframe(
                                    significant_pathways.round(6),
                                    use_container_width=True
                                )
                                
                                # Create bar plot
                                self.create_pathway_plot(significant_pathways)
                            else:
                                st.warning("No significant pathways found (p < 0.05)")
                                st.dataframe(pathway_results.head(10))
                        else:
                            st.warning("No pathway enrichment results found.")
                            
                    except Exception as e:
                        st.error(f"Pathway analysis failed: {e}")
    
    def create_pathway_plot(self, pathway_results):
        """Create pathway enrichment plot"""
        try:
            px, go, make_subplots, sns, plt = get_plotting_libs()
            if px is None:
                st.warning("Plotting libraries not available")
                return
            
            # Prepare data for plotting
            plot_data = pathway_results.head(10).copy()
            plot_data['-log10(P_value)'] = -np.log10(plot_data['P_value'] + 1e-300)
            
            # Create horizontal bar plot
            fig = px.bar(
                plot_data.sort_values('-log10(P_value)'),
                x='-log10(P_value)',
                y='Pathway',
                color='Enrichment_Score',
                title='Top Enriched Pathways',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=400,
                xaxis_title='-log10(P-value)',
                yaxis_title='Pathway'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create pathway plot: {e}")
    
    def literature_search_section(self, tab):
        """Literature search section"""
        with tab:
            st.header("📚 Literature Search")
            
            st.markdown("""
            **Note:** Literature search requires internet access and external APIs.
            This section provides a framework for PubMed literature search.
            """)
            
            # Search interface
            st.subheader("🔍 Search Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_source = st.selectbox(
                    "Search source:",
                    ["Manual entry", "From DE analysis", "From pathway analysis"]
                )
            
            with col2:
                max_results = st.number_input("Maximum results:", 5, 100, 20, 5)
            
            # Gene/term selection
            search_terms = []
            
            if search_source == "Manual entry":
                manual_terms = st.text_area(
                    "Enter search terms (one per line):",
                    placeholder="BRCA1\nTP53\napoptosis\ncancer"
                )
                search_terms = [term.strip() for term in manual_terms.split('\n') if term.strip()]
                
            elif search_source == "From DE analysis":
                if 'differential_expression' in st.session_state.analysis_results:
                    de_results = st.session_state.analysis_results['differential_expression']
                    top_genes = de_results[de_results['Significant']]['Gene'].head(10).tolist()
                    search_terms = st.multiselect("Select genes to search:", top_genes, default=top_genes[:5])
                else:
                    st.warning("No differential expression results available.")
                    
            elif search_source == "From pathway analysis":
                if 'pathway_analysis' in st.session_state.analysis_results:
                    pathway_results = st.session_state.analysis_results['pathway_analysis']
                    top_pathways = pathway_results['Pathway'].head(5).tolist()
                    search_terms = st.multiselect("Select pathways to search:", top_pathways, default=top_pathways[:3])
                else:
                    st.warning("No pathway analysis results available.")
            
            # Mock search results (since we can't access PubMed directly)
            if st.button("🔍 Search Literature") and search_terms:
                st.subheader("📄 Search Results (Mock)")
                
                for term in search_terms[:3]:  # Limit to 3 terms
                    with st.expander(f"📚 Results for: {term}"):
                        st.markdown(f"""
                        **Mock Literature Results for {term}:**
                        
                        1. **Title:** "Functional analysis of {term} in cancer progression"
                           - **Authors:** Smith J, et al.
                           - **Journal:** Nature Genetics (2023)
                           - **PMID:** 12345678
                           - **Abstract:** This study investigates the role of {term} in cancer development and progression...
                        
                        2. **Title:** "Therapeutic targeting of {term} pathway in oncology"
                           - **Authors:** Johnson A, et al.
                           - **Journal:** Cell (2023)
                           - **PMID:** 87654321
                           - **Abstract:** We demonstrate that targeting {term} represents a promising therapeutic approach...
                        
                        *Note: These are mock results. Real implementation would query PubMed API.*
                        """)
                
                st.info("💡 To implement real literature search, add PubMed API integration with the `pymed` package.")
    
    def advanced_visualizations_section(self, tab):
        """Advanced visualizations section"""
        with tab:
            st.header("🎨 Advanced Visualizations")
            
            if st.session_state.expression_data is None:
                st.warning("Please load expression data first!")
                return
            
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Heatmap", "PCA Plot", "Box Plots", "Correlation Matrix", "Expression Distribution"]
            )
            
            if viz_type == "Heatmap":
                self.create_heatmap_section()
            elif viz_type == "PCA Plot":
                self.create_pca_section()
            elif viz_type == "Box Plots":
                self.create_boxplot_section()
            elif viz_type == "Correlation Matrix":
                self.create_correlation_section()
            elif viz_type == "Expression Distribution":
                self.create_distribution_section()
    
    def create_heatmap_section(self):
        """Create heatmap visualization"""
        st.subheader("🔥 Expression Heatmap")
        
        expr_data = st.session_state.expression_data
        
        # Gene selection
        col1, col2 = st.columns(2)
        with col1:
            gene_selection = st.selectbox(
                "Gene selection:",
                ["Top variable genes", "From DE analysis", "Custom list"]
            )
        
        with col2:
            n_genes = st.slider("Number of genes:", 10, 100, 50, 10)
        
        if gene_selection == "Top variable genes":
            # Select most variable genes
            gene_vars = expr_data.var(axis=1)
            top_genes = gene_vars.nlargest(n_genes).index
            plot_data = expr_data.loc[top_genes]
            
        elif gene_selection == "From DE analysis":
            if 'differential_expression' in st.session_state.analysis_results:
                de_results = st.session_state.analysis_results['differential_expression']
                top_genes = de_results.nsmallest(n_genes, 'padj')['Gene'].tolist()
                plot_data = expr_data.loc[top_genes]
            else:
                st.warning("No DE analysis results available.")
                return
                
        else:  # Custom list
            gene_input = st.text_area("Enter gene names (one per line):")
            genes = [g.strip() for g in gene_input.split('\n') if g.strip()]
            if genes:
                available_genes = [g for g in genes if g in expr_data.index]
                if available_genes:
                    plot_data = expr_data.loc[available_genes]
                else:
                    st.warning("No matching genes found.")
                    return
            else:
                st.warning("Please enter gene names.")
                return
        
        if st.button("Generate Heatmap"):
            try:
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px is None:
                    st.warning("Plotting libraries not available")
                    return
                
                # Normalize data (z-score)
                from scipy.stats import zscore
                plot_data_norm = plot_data.apply(zscore, axis=1)
                
                # Create heatmap
                fig = px.imshow(
                    plot_data_norm,
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title=f'Expression Heatmap ({len(plot_data)} genes)'
                )
                
                fig.update_layout(
                    xaxis_title='Samples',
                    yaxis_title='Genes',
                    height=max(400, len(plot_data) * 15)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to create heatmap: {e}")
    
    def create_pca_section(self):
        """Create PCA visualization"""
        st.subheader("📊 Principal Component Analysis")
        
        if 'pca' not in self.libs:
            st.warning("PCA requires scikit-learn. Install with: pip install scikit-learn")
            return
        
        expr_data = st.session_state.expression_data
        
        # PCA parameters
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Number of components:", 2, 10, 2, 1)
        with col2:
            scale_data = st.checkbox("Scale data", value=True)
        
        if st.button("Run PCA"):
            try:
                PCA = self.libs['pca']
                StandardScaler = self.libs['scaler']
                
                # Prepare data (samples as rows, genes as columns)
                data_for_pca = expr_data.T
                
                if scale_data:
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_for_pca)
                else:
                    data_scaled = data_for_pca.values
                
                # Run PCA
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(data_scaled)
                
                # Create DataFrame
                pca_df = pd.DataFrame(
                    pca_result,
                    index=data_for_pca.index,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
                
                # Add clinical data if available
                if st.session_state.clinical_data is not None:
                    common_samples = set(pca_df.index) & set(st.session_state.clinical_data.index)
                    pca_df_subset = pca_df.loc[list(common_samples)]
                    clinical_subset = st.session_state.clinical_data.loc[list(common_samples)]
                    
                    # Color by first categorical column
                    color_col = None
                    for col in clinical_subset.columns:
                        if clinical_subset[col].dtype == 'object' or clinical_subset[col].nunique() < 10:
                            color_col = col
                            break
                    
                    if color_col:
                        pca_df_subset[color_col] = clinical_subset[color_col]
                else:
                    pca_df_subset = pca_df
                    color_col = None
                
                # Plot PCA
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px:
                    fig = px.scatter(
                        pca_df_subset,
                        x='PC1',
                        y='PC2',
                        color=color_col,
                        title=f'PCA Plot (PC1 vs PC2)',
                        hover_name=pca_df_subset.index
                    )
                    
                    fig.update_layout(
                        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show explained variance
                st.subheader("📈 Explained Variance")
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Explained_Variance_Ratio': pca.explained_variance_ratio_,
                    'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
                })
                st.dataframe(variance_df)
                
            except Exception as e:
                st.error(f"PCA analysis failed: {e}")
    
    def create_boxplot_section(self):
        """Create box plot visualization"""
        st.subheader("📦 Expression Box Plots")
        
        expr_data = st.session_state.expression_data
        
        # Gene selection
        available_genes = list(expr_data.index)
        selected_genes = st.multiselect(
            "Select genes to plot:",
            available_genes,
            default=available_genes[:5] if len(available_genes) >= 5 else available_genes
        )
        
        if selected_genes and st.button("Generate Box Plots"):
            try:
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px is None:
                    st.warning("Plotting libraries not available")
                    return
                
                # Prepare data for plotting
                plot_data = []
                for gene in selected_genes:
                    for sample, value in expr_data.loc[gene].items():
                        plot_data.append({
                            'Gene': gene,
                            'Sample': sample,
                            'Expression': value
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                # Add clinical data if available
                if st.session_state.clinical_data is not None:
                    # Merge with clinical data
                    clinical_data = st.session_state.clinical_data.reset_index()
                    clinical_data.rename(columns={'index': 'Sample'}, inplace=True)
                    plot_df = plot_df.merge(clinical_data, on='Sample', how='left')
                
                # Create box plot
                fig = px.box(
                    plot_df,
                    x='Gene',
                    y='Expression',
                    title='Gene Expression Box Plots'
                )
                
                fig.update_layout(
                    xaxis_title='Genes',
                    yaxis_title='Expression Level'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to create box plots: {e}")
    
    def create_correlation_section(self):
        """Create correlation matrix"""
        st.subheader("🔗 Sample Correlation Matrix")
        
        expr_data = st.session_state.expression_data
        
        # Correlation parameters
        col1, col2 = st.columns(2)
        with col1:
            corr_method = st.selectbox("Correlation method:", ["pearson", "spearman"])
        with col2:
            n_samples = st.slider("Max samples to show:", 10, min(50, len(expr_data.columns)), 
                                 min(20, len(expr_data.columns)), 5)
        
        if st.button("Generate Correlation Matrix"):
            try:
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px is None:
                    st.warning("Plotting libraries not available")
                    return
                
                # Calculate correlation
                subset_data = expr_data.iloc[:, :n_samples]
                corr_matrix = subset_data.corr(method=corr_method)
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title=f'Sample Correlation Matrix ({corr_method})'
                )
                
                fig.update_layout(
                    xaxis_title='Samples',
                    yaxis_title='Samples'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                st.write("**Correlation Statistics:**")
                st.write(f"Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
                st.write(f"Median correlation: {np.median(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]):.3f}")
                
            except Exception as e:
                st.error(f"Failed to create correlation matrix: {e}")
    
    def create_distribution_section(self):
        """Create expression distribution plots"""
        st.subheader("📊 Expression Distributions")
        
        expr_data = st.session_state.expression_data
        
        dist_type = st.selectbox(
            "Distribution type:",
            ["Overall distribution", "Per-sample distribution", "Per-gene distribution"]
        )
        
        if st.button("Generate Distribution Plot"):
            try:
                px, go, make_subplots, sns, plt = get_plotting_libs()
                if px is None:
                    st.warning("Plotting libraries not available")
                    return
                
                if dist_type == "Overall distribution":
                    # Flatten all expression values
                    all_values = expr_data.values.flatten()
                    
                    fig = px.histogram(
                        x=all_values,
                        nbins=50,
                        title='Overall Expression Distribution'
                    )
                    fig.update_layout(
                        xaxis_title='Expression Level',
                        yaxis_title='Frequency'
                    )
                    
                elif dist_type == "Per-sample distribution":
                    # Box plot of expression per sample
                    sample_data = []
                    for sample in expr_data.columns[:20]:  # Limit to first 20 samples
                        for value in expr_data[sample]:
                            sample_data.append({
                                'Sample': sample,
                                'Expression': value
                            })
                    
                    sample_df = pd.DataFrame(sample_data)
                    fig = px.box(
                        sample_df,
                        x='Sample',
                        y='Expression',
                        title='Expression Distribution per Sample'
                    )
                    fig.update_xaxes(tickangle=45)
                    
                else:  # Per-gene distribution
                    # Select top variable genes
                    gene_vars = expr_data.var(axis=1)
                    top_genes = gene_vars.nlargest(10).index
                    
                    gene_data = []
                    for gene in top_genes:
                        for value in expr_data.loc[gene]:
                            gene_data.append({
                                'Gene': gene,
                                'Expression': value
                            })
                    
                    gene_df = pd.DataFrame(gene_data)
                    fig = px.box(
                        gene_df,
                        x='Gene',
                        y='Expression',
                        title='Expression Distribution per Gene (Top 10 Variable)'
                    )
                    fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to create distribution plot: {e}")
    
    def export_results_section(self, tab):
        """Export and download section"""
        with tab:
            st.header("💾 Export Results")
            
            if not st.session_state.analysis_results:
                st.warning("No analysis results to export. Please run some analyses first.")
                return
            
            st.subheader("📋 Available Results")
            
            # Show available results
            for result_name, result_data in st.session_state.analysis_results.items():
                if isinstance(result_data, pd.DataFrame):
                    st.write(f"✅ **{result_name.replace('_', ' ').title()}**: {len(result_data)} rows")
                else:
                    st.write(f"✅ **{result_name.replace('_', ' ').title()}**: Available")
            
            # Export options
            st.subheader("💾 Export Options")
            
            export_format = st.selectbox(
                "Select export format:",
                ["Excel (.xlsx)", "CSV", "JSON", "Summary Report"]
            )
            
            if st.button("📦 Generate Export Package"):
                try:
                    if export_format == "Excel (.xlsx)":
                        # Create Excel file with multiple sheets
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            for result_name, result_data in st.session_state.analysis_results.items():
                                if isinstance(result_data, pd.DataFrame):
                                    result_data.to_excel(writer, sheet_name=result_name[:31])  # Excel sheet name limit
                        
                        st.download_button(
                            label="📥 Download Excel File",
                            data=output.getvalue(),
                            file_name=f"prairie_genomics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    elif export_format == "CSV":
                        # Export each result as separate CSV
                        for result_name, result_data in st.session_state.analysis_results.items():
                            if isinstance(result_data, pd.DataFrame):
                                csv = result_data.to_csv(index=True)
                                st.download_button(
                                    label=f"📥 Download {result_name.title()} CSV",
                                    data=csv,
                                    file_name=f"{result_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                    
                    elif export_format == "JSON":
                        # Convert results to JSON
                        json_data = {}
                        for result_name, result_data in st.session_state.analysis_results.items():
                            if isinstance(result_data, pd.DataFrame):
                                json_data[result_name] = result_data.to_dict('records')
                        
                        json_str = json.dumps(json_data, indent=2, default=str)
                        st.download_button(
                            label="📥 Download JSON File",
                            data=json_str,
                            file_name=f"prairie_genomics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    elif export_format == "Summary Report":
                        # Generate summary report
                        report = self.generate_summary_report()
                        st.download_button(
                            label="📥 Download Summary Report",
                            data=report,
                            file_name=f"prairie_genomics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    st.success("✅ Export package generated successfully!")
                    
                except Exception as e:
                    st.error(f"Export failed: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        report = []
        report.append("=" * 60)
        report.append("PRAIRIE GENOMICS SUITE - ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data summary
        if st.session_state.expression_data is not None:
            expr_shape = st.session_state.expression_data.shape
            report.append(f"EXPRESSION DATA: {expr_shape[0]} genes × {expr_shape[1]} samples")
        
        if st.session_state.clinical_data is not None:
            clin_shape = st.session_state.clinical_data.shape
            report.append(f"CLINICAL DATA: {clin_shape[0]} samples × {clin_shape[1]} variables")
        
        report.append("")
        report.append("ANALYSIS RESULTS:")
        report.append("-" * 40)
        
        # Analysis summaries
        for result_name, result_data in st.session_state.analysis_results.items():
            report.append(f"\n{result_name.upper().replace('_', ' ')}:")
            
            if isinstance(result_data, pd.DataFrame):
                if result_name == "differential_expression":
                    n_significant = result_data['Significant'].sum() if 'Significant' in result_data.columns else 0
                    n_up = ((result_data['log2FoldChange'] > 0) & result_data['Significant']).sum() if 'Significant' in result_data.columns else 0
                    n_down = ((result_data['log2FoldChange'] < 0) & result_data['Significant']).sum() if 'Significant' in result_data.columns else 0
                    
                    report.append(f"  - Total genes analyzed: {len(result_data)}")
                    report.append(f"  - Significant genes: {n_significant}")
                    report.append(f"  - Upregulated: {n_up}")
                    report.append(f"  - Downregulated: {n_down}")
                
                elif result_name == "pathway_analysis":
                    n_pathways = len(result_data)
                    n_significant = (result_data['P_value'] < 0.05).sum() if 'P_value' in result_data.columns else 0
                    
                    report.append(f"  - Total pathways tested: {n_pathways}")
                    report.append(f"  - Significant pathways (p<0.05): {n_significant}")
                    
                    if n_significant > 0:
                        top_pathway = result_data.loc[result_data['P_value'].idxmin(), 'Pathway']
                        report.append(f"  - Top pathway: {top_pathway}")
                
                else:
                    report.append(f"  - Records: {len(result_data)}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        
        return "\n".join(report)
    
    def settings_section(self, tab):
        """Application settings section"""
        with tab:
            st.header("⚙️ Settings & Configuration")
            
            # Analysis settings
            st.subheader("🔧 Analysis Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_p_threshold = st.number_input(
                    "P-value threshold:",
                    0.001, 0.1, 
                    st.session_state.settings['p_threshold'], 
                    0.001
                )
            
            with col2:
                new_fc_threshold = st.number_input(
                    "Fold change threshold:",
                    1.1, 5.0,
                    st.session_state.settings['fc_threshold'],
                    0.1
                )
            
            with col3:
                new_min_expression = st.number_input(
                    "Minimum expression:",
                    0.0, 10.0,
                    st.session_state.settings['min_expression'],
                    0.1
                )
            
            if st.button("💾 Save Settings"):
                st.session_state.settings.update({
                    'p_threshold': new_p_threshold,
                    'fc_threshold': new_fc_threshold,
                    'min_expression': new_min_expression
                })
                st.success("Settings saved!")
            
            # System information
            st.subheader("🖥️ System Information")
            
            # Available libraries
            st.write("**Available Libraries:**")
            lib_status = {
                "Pandas": "✅",
                "NumPy": "✅", 
                "SciPy": "✅" if 'scipy_stats' in self.libs else "❌",
                "Scikit-learn": "✅" if 'pca' in self.libs else "❌",
                "MyGene": "✅" if 'mygene' in self.libs else "❌",
                "GSEAPy": "✅" if 'gseapy' in self.libs else "❌",
                "Lifelines": "✅ (assumed)" if True else "❌",
                "Plotly": "✅ (assumed)" if True else "❌"
            }
            
            for lib, status in lib_status.items():
                st.write(f"- {lib}: {status}")
            
            # Performance tips
            st.subheader("⚡ Performance Tips")
            st.markdown("""
            **For optimal performance:**
            
            1. **Data Size**: Keep expression data under 50MB for smooth operation
            2. **Gene Filtering**: Filter low-expression genes before analysis
            3. **Sample Size**: Large sample sizes (>100) may slow down some analyses
            4. **Browser**: Use Chrome or Firefox for best Plotly visualization performance
            5. **Memory**: Close other browser tabs if experiencing slowdowns
            
            **Troubleshooting:**
            - If plots don't appear: Refresh the page
            - If analysis fails: Check data format and try example data
            - For large datasets: Use the filtering options to reduce data size
            """)
            
            # Reset options
            st.subheader("🔄 Reset Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear Analysis Results"):
                    st.session_state.analysis_results = {}
                    st.success("Analysis results cleared!")
            
            with col2:
                if st.button("🔄 Reset All Settings"):
                    st.session_state.settings = {
                        'p_threshold': 0.05,
                        'fc_threshold': 1.5,
                        'min_expression': 1.0
                    }
                    st.success("Settings reset to defaults!")
    
    # ================================
    # MAIN APPLICATION RUNNER
    # ================================
    
    def run(self):
        """Run the main application"""
        # Show sidebar
        self.show_sidebar()
        
        # Main header and navigation
        tabs = self.show_header()
        
        # Run each section in its respective tab
        self.data_import_section(tabs[0])
        self.gene_conversion_section(tabs[1])
        self.differential_expression_section(tabs[2])
        self.survival_analysis_section(tabs[3])
        self.pathway_analysis_section(tabs[4])
        self.literature_search_section(tabs[5])
        self.advanced_visualizations_section(tabs[6])
        self.export_results_section(tabs[7])
        self.settings_section(tabs[8])
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;'>
            <h3 style='color: #2c3e50;'>🧬 Prairie Genomics Suite</h3>
            <p style='margin: 0.5rem 0;'><strong>Streamlit Cloud Ready Edition</strong></p>
            <p style='margin: 0.5rem 0;'>Publication-ready genomics analysis platform</p>
            <p style='margin: 0.5rem 0; font-size: 0.9em;'>Built with Streamlit • Python • Advanced Analytics</p>
            <p style='margin: 0; font-style: italic; color: #7f8c8d;'>Making genomics analysis accessible to every researcher</p>
            <hr style='margin: 1rem 0; border: none; height: 1px; background-color: #ddd;'>
            <p style='margin: 0; font-size: 0.8em; color: #95a5a6;'>
                Version 2.1.0 | Optimized for Streamlit Cloud | 
                <a href='https://github.com/prairie-genomics/suite' style='color: #3498db;'>GitHub</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================================
# MAIN APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    # Initialize and run the Streamlit-ready application
    try:
        app = PrairieGenomicsStreamlit()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        st.info("Please check your Python environment and dependencies.")