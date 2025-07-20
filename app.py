import streamlit as st
import json
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from transformers import pipeline

# Set page config
st.set_page_config(
    page_title="JSON Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'structured_data' not in st.session_state:
    st.session_state.structured_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

class JSONDataProcessor:
    def __init__(self):
        self.raw_data = None
        self.structured_data = None
        self.metadata = {}
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded JSON files"""
        try:
            if len(uploaded_files) == 1:
                self.raw_data = json.load(uploaded_files[0])
            else:
                self.raw_data = [json.load(file) for file in uploaded_files]
            
            self._convert_to_structured_data()
            self._clean_data()
            self._extract_metadata()
            
            st.session_state.uploaded_data = self.raw_data
            st.session_state.structured_data = self.structured_data
            
            return True
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            return False
    
    def _convert_to_structured_data(self):
        """Convert JSON to structured format"""
        if isinstance(self.raw_data, list):
            self.structured_data = pd.json_normalize(self.raw_data)
        else:
            self.structured_data = pd.json_normalize([self.raw_data])
    
    def _clean_data(self):
        """Handle missing values and data types"""
        # Fill NA/NaN values
        self.structured_data = self.structured_data.fillna('MISSING')
        
        # Convert numeric columns
        for col in self.structured_data.columns:
            try:
                self.structured_data[col] = pd.to_numeric(self.structured_data[col])
            except ValueError:
                pass
    
    def _extract_metadata(self):
        """Extract metadata about the dataset"""
        self.metadata = {
            'columns': list(self.structured_data.columns),
            'num_records': len(self.structured_data),
            'data_types': str(self.structured_data.dtypes.to_dict()),
            'sample_records': self.structured_data.head().to_dict('records')
        }

class DataAnalyzer:
    def __init__(self, structured_data):
        self.data = structured_data
        self.analysis_results = {}
    
    def analyze_common_terms(self, text_columns=None):
        """Find most common terms in text columns"""
        if text_columns is None:
            text_columns = self._identify_text_columns()
        
        term_results = {}
        for col in text_columns:
            if col in self.data.columns:
                all_text = ' '.join(self.data[col].astype(str).values)
                words = all_text.lower().split()
                word_counts = Counter(words)
                common_terms = word_counts.most_common(20)
                term_results[col] = common_terms
        
        self.analysis_results['common_terms'] = term_results
        st.session_state.analysis_results = self.analysis_results
        return term_results
    
    def _identify_text_columns(self):
        """Identify columns likely containing text"""
        text_cols = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object' and \
               any(isinstance(x, str) and ' ' in x for x in self.data[col].head().values):
                text_cols.append(col)
        return text_cols
    
    def find_correlations(self):
        """Calculate correlations between numeric columns"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            correlations = numeric_data.corr()
            self.analysis_results['correlations'] = correlations
            st.session_state.analysis_results = self.analysis_results
            return correlations
        return None
    
    def generate_summary_stats(self):
        """Generate descriptive statistics"""
        stats = {
            'descriptive': self.data.describe().to_dict(),
            'missing_values': self.data.isnull().sum().to_dict()
        }
        self.analysis_results['summary_stats'] = stats
        st.session_state.analysis_results = self.analysis_results
        return stats

class ModelTester:
    def __init__(self, structured_data):
        self.data = structured_data
        self.models = {}
        self.results = {}
    
    def prepare_data(self, target_column, test_size=0.2):
        """Prepare data for modeling"""
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        clean_data = self.data.dropna(subset=[target_column])
        X = clean_data.drop(columns=[target_column])
        y = clean_data[target_column]
        X = pd.get_dummies(X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=42)
        
        return X.shape
    
    def test_regression_model(self):
        """Test linear regression model"""
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.results['linear_regression'] = {
            'model': model,
            'metrics': {'mse': mse, 'r2_score': r2},
            'coefficients': dict(zip(self.X_train.columns, model.coef_))
        }
        
        st.session_state.model_results = self.results
        return self.results['linear_regression']
    
    def test_tree_model(self):
        """Test decision tree model"""
        model = DecisionTreeRegressor(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.results['decision_tree'] = {
            'model': model,
            'metrics': {'mse': mse, 'r2_score': r2},
            'feature_importances': dict(zip(self.X_train.columns, model.feature_importances_))
        }
        
        st.session_state.model_results = self.results
        return self.results['decision_tree']
    
    def test_random_forest(self):
        """Test random forest model"""
        model = RandomForestRegressor(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.results['random_forest'] = {
            'model': model,
            'metrics': {'mse': mse, 'r2_score': r2},
            'feature_importances': dict(zip(self.X_train.columns, model.feature_importances_))
        }
        
        st.session_state.model_results = self.results
        return self.results['random_forest']
    
    def test_llm_analysis(self, text_column, question):
        """Use LLM to analyze text data"""
        if text_column not in self.data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )
        
        context = ' '.join(self.data[text_column].astype(str).values[:1000])
        result = qa_pipeline(question=question, context=context)
        
        self.results['llm_analysis'] = {
            'question': question,
            'answer': result['answer'],
            'score': result['score']
        }
        
        st.session_state.model_results = self.results
        return self.results['llm_analysis']

def plot_common_terms(common_terms):
    """Visualize most common terms"""
    for col, terms in common_terms.items():
        words, counts = zip(*terms)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words), ax=ax)
        ax.set_title(f"Most Common Terms in '{col}'")
        ax.set_xlabel("Frequency")
        st.pyplot(fig)

def plot_correlations(correlations):
    """Visualize correlation matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlations")
    st.pyplot(fig)

def plot_feature_importance(feature_importances, model_name):
    """Visualize feature importance for a model"""
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    features, importance_values = zip(*sorted_importances)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(importance_values), y=list(features), ax=ax)
    ax.set_title(f"Feature Importance - {model_name}")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

def create_relationship_graph(correlations):
    """Create a graph showing relationships between entities"""
    G = nx.Graph()
    
    if correlations is not None:
        for col in correlations.columns:
            G.add_node(col, type='feature')
        
        for i, col1 in enumerate(correlations.columns):
            for j, col2 in enumerate(correlations.columns):
                if i < j and abs(correlations.iloc[i, j]) > 0.5:
                    G.add_edge(col1, col2, weight=correlations.iloc[i, j])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    ax.set_title("Data Feature Relationships")
    st.pyplot(fig)

# Main App
def main():
    st.title("📊 JSON Data Analyzer")
    st.markdown("Upload JSON files to analyze data and test machine learning models")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload JSON Files")
        uploaded_files = st.file_uploader(
            "Choose JSON files",
            type=["json"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            processor = JSONDataProcessor()
            if processor.process_uploaded_files(uploaded_files):
                st.success("Files processed successfully!")
                st.session_state.processor = processor
            else:
                st.error("Failed to process files")
    
    # Main content
    if st.session_state.get('structured_data') is not None:
        data = st.session_state.structured_data
        analyzer = DataAnalyzer(data)
        model_tester = ModelTester(data)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Data Analysis", "Model Testing", "LLM Analysis"])
        
        with tab1:
            st.header("Data Overview")
            st.write(f"**Number of records:** {len(data)}")
            st.write(f"**Number of columns:** {len(data.columns)}")
            
            st.subheader("Sample Data")
            st.dataframe(data.head())
            
            st.subheader("Column Information")
            st.write(data.dtypes)
            
            st.subheader("Missing Values")
            st.write(data.isnull().sum())
        
        with tab2:
            st.header("Data Analysis")
            
            if st.button("Run Full Analysis"):
                with st.spinner("Analyzing data..."):
                    common_terms = analyzer.analyze_common_terms()
                    correlations = analyzer.find_correlations()
                    summary_stats = analyzer.generate_summary_stats()
                
                st.subheader("Most Common Terms")
                if 'common_terms' in st.session_state.analysis_results:
                    plot_common_terms(st.session_state.analysis_results['common_terms'])
                
                st.subheader("Feature Correlations")
                if 'correlations' in st.session_state.analysis_results:
                    plot_correlations(st.session_state.analysis_results['correlations'])
                    create_relationship_graph(st.session_state.analysis_results.get('correlations'))
                
                st.subheader("Summary Statistics")
                if 'summary_stats' in st.session_state.analysis_results:
                    st.write(st.session_state.analysis_results['summary_stats']['descriptive'])
        
        with tab3:
            st.header("Model Testing")
            
            target_col = st.selectbox(
                "Select target column for modeling",
                options=data.select_dtypes(include=[np.number]).columns,
                index=0 if len(data.select_dtypes(include=[np.number]).columns) > 0 else None
            )
            
            if target_col:
                if st.button("Test All Models"):
                    with st.spinner("Training models..."):
                        model_tester.prepare_data(target_col)
                        regression_results = model_tester.test_regression_model()
                        tree_results = model_tester.test_tree_model()
                        forest_results = model_tester.test_random_forest()
                    
                    st.subheader("Linear Regression Results")
                    st.write("Metrics:", regression_results['metrics'])
                    
                    st.subheader("Decision Tree Results")
                    st.write("Metrics:", tree_results['metrics'])
                    plot_feature_importance(tree_results['feature_importances'], "Decision Tree")
                    
                    st.subheader("Random Forest Results")
                    st.write("Metrics:", forest_results['metrics'])
                    plot_feature_importance(forest_results['feature_importances'], "Random Forest")
        
        with tab4:
            st.header("LLM Analysis")
            
            text_col = st.selectbox(
                "Select text column for analysis",
                options=data.select_dtypes(include=['object']).columns,
                index=0 if len(data.select_dtypes(include=['object']).columns) > 0 else None
            )
            
            if text_col:
                question = st.text_input("Enter your question about the data")
                
                if question and st.button("Get Answer"):
                    with st.spinner("Analyzing with LLM..."):
                        result = model_tester.test_llm_analysis(text_col, question)
                    
                    st.subheader("LLM Answer")
                    st.write(f"**Question:** {result['question']}")
                    st.write(f"**Answer:** {result['answer']}")
                    st.write(f"**Confidence Score:** {result['score']:.2f}")
    
    else:
        st.info("Please upload JSON files to begin analysis")

if __name__ == "__main__":
    main()
