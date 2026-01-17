import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts() -> Tuple:
    """Load the trained model and scaler from disk."""
    import os
    try:
        # Try to load from current directory first
        if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
            model = joblib.load('best_model.pkl')
            scaler = joblib.load('scaler.pkl')
        else:
            # Try alternative path for nested directory structure
            alt_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
            if os.path.exists(alt_path):
                model = joblib.load(alt_path)
                scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.pkl'))
            else:
                return None, None, f"Model files not found in current directory or {os.path.dirname(__file__)}"
        return model, scaler, None
    except FileNotFoundError as e:
        error_msg = f"Model files not found: {str(e)}"
        return None, None, error_msg
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        return None, None, error_msg


def validate_input(value: float, min_val: float, max_val: float, name: str) -> Tuple[bool, str]:
    """Validate input values are within acceptable ranges."""
    if value < min_val or value > max_val:
        return False, f"{name} must be between {min_val} and {max_val}"
    return True, ""


def preprocess_input(input_data: Dict[str, float], scaler) -> np.ndarray:
    """
    Preprocess input data following the exact pipeline from Analysis.ipynb.
    Feature order: cpu_usage, memory_usage, request_count, latency, error_count
    """
    # Create DataFrame with exact feature order from training
    feature_order = ['cpu_usage', 'memory_usage', 'request_count', 'latency', 'error_count']
    df = pd.DataFrame([input_data], columns=feature_order)
    
    # Apply scaling (StandardScaler fitted during training)
    scaled_data = scaler.transform(df)
    
    return scaled_data


def predict_failure(model, scaler, input_data: Dict[str, float], threshold: float = 0.4) -> Dict:
    """
    Make prediction using the loaded model with configurable threshold.
    Returns prediction details including probabilities.
    """
    # Preprocess input
    X = preprocess_input(input_data, scaler)
    
    # Get decision function scores (for Logistic Regression)
    decision_scores = model.decision_function(X)
    
    # Convert to probability using sigmoid
    probability = 1 / (1 + np.exp(-decision_scores[0]))
    
    # Apply threshold
    prediction = 1 if probability >= threshold else 0
    
    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': probability if prediction == 1 else (1 - probability),
        'threshold': threshold
    }


def create_probability_gauge(probability: float, threshold: float):
    """Create a gauge chart showing failure probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Failure Probability (%)", 'font': {'size': 20}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': '#d4edda'},
                {'range': [threshold * 100, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_feature_comparison(input_data: Dict[str, float]):
    """Create a bar chart comparing input features to typical ranges."""
    typical_ranges = {
        'cpu_usage': 50,
        'memory_usage': 60,
        'request_count': 500,
        'latency': 100,
        'error_count': 5
    }
    
    features = list(input_data.keys())
    input_values = [input_data[f] for f in features]
    typical_values = [typical_ranges[f] for f in features]
    
    fig = go.Figure(data=[
        go.Bar(name='Your Input', x=features, y=input_values, marker_color='#1f77b4'),
        go.Bar(name='Typical Values', x=features, y=typical_values, marker_color='#90EE90', opacity=0.6)
    ])
    
    fig.update_layout(
        barmode='group',
        title='Input Values vs. Typical System Behavior',
        xaxis_title='Metrics',
        yaxis_title='Values',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def main():
    # Initialize session state for prediction results
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = None
    
    # Load model artifacts
    model, scaler, error = load_model_artifacts()
    
    # Header
    st.markdown('<p class="main-header">🔧 Predictive Maintenance System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Early-Warning System for Hardware/Software Failures</p>', unsafe_allow_html=True)
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("Please ensure 'best_model.pkl' and 'scaler.pkl' are in the same directory as this application.")
        st.stop()
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown("""
        **Model Type:** Logistic Regression  
        **Optimization:** Recall-Focused (Failure Detection)  
        **Training Data:** 876,100 system logs  
        **Failure Rate:** 0.08% (Extreme Imbalance)
        """)
        
        st.divider()
        
        st.header("⚙️ Configuration")
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.1,
            max_value=0.5,
            value=0.4,
            step=0.1,
            help="Lower threshold = Higher sensitivity to failures"
        )
        
        st.markdown(f"""
        **Current Setting:** {threshold}
        
        **Recommended Thresholds:**
        - **0.3-0.4:** Balanced (Recommended)
        - **0.2:** Aggressive (High Recall)
        - **0.1:** Extreme (Catches All)
        """)
        
        st.divider()
        
        st.header("📊 Dataset Features")
        st.markdown("""
        - CPU Usage (%)
        - Memory Usage (%)
        - Request Count
        - Latency (ms)
        - Error Count
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📖 Overview", "🔬 Model Details", "⚠️ Limitations"])
    
    with tab1:
        st.header("System Health Assessment")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input System Metrics")
            
            # Input form
            with st.form("prediction_form"):
                cpu_usage = st.number_input(
                    "CPU Usage (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=45.0,
                    step=1.0,
                    help="Percentage of CPU utilization"
                )
                
                memory_usage = st.number_input(
                    "Memory Usage (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=55.0,
                    step=1.0,
                    help="Percentage of memory in use"
                )
                
                request_count = st.number_input(
                    "Request Count",
                    min_value=0,
                    max_value=10000,
                    value=450,
                    step=10,
                    help="Total requests processed per interval"
                )
                
                latency = st.number_input(
                    "Latency (ms)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=95.0,
                    step=5.0,
                    help="System response time in milliseconds"
                )
                
                error_count = st.number_input(
                    "Error Count",
                    min_value=0,
                    max_value=1000,
                    value=3,
                    step=1,
                    help="Number of errors logged in the interval"
                )
                
                submit_button = st.form_submit_button("🔍 Analyze System Health", use_container_width=True)
            
            if submit_button:
                # Prepare input data
                input_data = {
                    'cpu_usage': float(cpu_usage),
                    'memory_usage': float(memory_usage),
                    'request_count': float(request_count),
                    'latency': float(latency),
                    'error_count': float(error_count)
                }
                
                # Make prediction
                result = predict_failure(model, scaler, input_data, threshold)
                
                # Store in session state
                st.session_state['prediction_result'] = result
                st.session_state['input_data'] = input_data
        
        with col2:
            st.subheader("Analysis Results")
            
            if st.session_state['prediction_result'] is not None:
                result = st.session_state['prediction_result']
                input_data = st.session_state['input_data']
                
                # Display gauge
                fig = create_probability_gauge(result['probability'], result['threshold'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction outcome
                if result['prediction'] == 1:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3 style="color: #dc3545; margin-top: 0;">⚠️ FAILURE RISK DETECTED</h3>
                        <p><strong>Recommendation:</strong> Immediate maintenance required</p>
                        <p><strong>Confidence:</strong> {result['confidence']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="color: #28a745; margin-top: 0;">✅ SYSTEM HEALTHY</h3>
                        <p><strong>Status:</strong> No immediate action required</p>
                        <p><strong>Confidence:</strong> {result['confidence']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feature comparison chart
                st.subheader("Metric Analysis")
                fig2 = create_feature_comparison(input_data)
                st.plotly_chart(fig2, use_container_width=True)
                
            else:
                st.info("👆 Enter system metrics and click 'Analyze System Health' to get predictions")
    
    with tab2:
        st.header("Project Overview")
        
        st.markdown("""
        ### 🎯 Goal
        This system provides an early-warning mechanism for identifying potential hardware or software 
        failures before they occur. By analyzing real-time system telemetry, we enable proactive 
        maintenance and reduce unplanned downtime.
        
        ### 📊 Dataset
        - **Total Records:** 876,100 system logs
        - **Failure Rate:** 0.08% (719 failures / 875,381 normal operations)
        - **Challenge:** Extreme class imbalance
        
        ### 🔧 Key Features Monitored
        1. **CPU Usage:** Processor utilization percentage
        2. **Memory Usage:** RAM consumption percentage  
        3. **Request Count:** System load indicator
        4. **Latency:** Response time performance
        5. **Error Count:** System stability metric
        
        ### ✨ Why This Matters
        - **Cost Savings:** Prevent expensive emergency repairs
        - **Uptime:** Minimize service disruptions
        - **Planning:** Schedule maintenance during optimal windows
        - **Safety:** Address issues before critical failures
        """)
    
    with tab3:
        st.header("Model Technical Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Model Performance")
            
            performance_data = pd.DataFrame({
                'Threshold': [0.5, 0.4, 0.3, 0.2, 0.1],
                'Recall': [0.5972, 0.8056, 0.9167, 0.9792, 1.0000],
                'F1-Score': [0.2960, 0.2886, 0.2573, 0.2384, 0.2255]
            })
            
            st.dataframe(performance_data, hide_index=True, use_container_width=True)
            
            st.markdown("""
            **Recall:** Percentage of actual failures correctly identified  
            **F1-Score:** Balance between precision and recall
            
            *Note: Performance measured on balanced test set*
            """)
        
        with col2:
            st.subheader("🔍 Methodology")
            
            st.markdown("""
            **Preprocessing:**
            - Feature scaling (StandardScaler)
            - Balanced sampling for training
            - Stratified train-test split
            
            **Model Selection:**
            - Algorithm: Logistic Regression
            - Optimization: `class_weight='balanced'`
            - Selection Criteria: Recall maximization
            
            **Why Logistic Regression?**
            - High threshold sensitivity
            - Stable on telemetry data
            - Interpretable decision boundaries
            - Outperformed Random Forest for minority class detection
            """)
        
        st.subheader("📉 Threshold Impact Visualization")
        
        fig = px.line(
            performance_data,
            x='Threshold',
            y=['Recall', 'F1-Score'],
            title='Model Performance Across Decision Thresholds',
            labels={'value': 'Score', 'variable': 'Metric'},
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("System Limitations & Considerations")
        
        st.markdown("""
        <div class="warning-box">
            <h3 style="margin-top: 0;">⚠️ Important Limitations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Model Scope
        - **Training Environment:** Model trained on balanced subset (719 failures + 5,000 normal cases)
        - **Real-World Imbalance:** Production environments will have significantly higher normal-to-failure ratios
        - **Performance Note:** Reported recall/F1 scores are from controlled test conditions
        
        ### 🔬 Technical Considerations
        
        **Class Imbalance:**
        - Original dataset: 0.08% failure rate (extreme imbalance)
        - Training used balanced sampling to prevent model from ignoring failures
        - Real deployment will see fewer failures per 1,000 predictions
        
        **False Positives:**
        - Lower thresholds (0.2-0.3) increase false alarms
        - Trade-off between catching all failures vs. unnecessary maintenance alerts
        - Recommended threshold (0.4) balances both concerns
        
        **Feature Dependencies:**
        - Model requires all 5 features for accurate predictions
        - Missing data will impact prediction quality
        - Feature scaling must match training distribution
        
        ### 💡 Best Practices
        
        1. **Threshold Selection:**
           - Use 0.4 for balanced operation (recommended)
           - Lower to 0.3 if failure costs are very high
           - Never use 0.5+ in production (defeats purpose)
        
        2. **Interpretation:**
           - Treat predictions as early warnings, not certainties
           - Combine with domain expertise and manual inspection
           - Consider maintenance history and context
        
        3. **Monitoring:**
           - Track false positive rates in production
           - Adjust threshold based on operational feedback
           - Retrain periodically with new failure data
        
        ### 🔄 Future Enhancements
        - Advanced resampling (SMOTE/ADASYN)
        - Hyperparameter optimization (Optuna)
        - Ensemble methods for improved robustness
        - Real-time streaming predictions
        
        ### 📝 Disclaimer
        This system is designed as a predictive aid, not a replacement for professional 
        maintenance judgment. Always verify alerts with qualified personnel before taking action.
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Predictive Maintenance System</strong> | Developed by Muhammad Khizar Arif | 2026</p>
        <p style="font-size: 0.9rem;">Model: Logistic Regression (Recall-Optimized) | Dataset: 876,100 System Logs</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()