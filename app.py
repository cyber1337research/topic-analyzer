import streamlit as st
import torch
from transformers import pipeline
import plotly.graph_objects as go

# Define available models
MODELS = {
    "mDeBERTa-v3-2mil7": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "mDeBERTa-v3-mnli": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "BART-large-mnli": "facebook/bart-large-mnli"
}

# Define example texts and categories
CATEGORIES = [
    "personal financial information",
    "company financial information",
    "human resources and employment",
    "legal consulting",
    "health and medical information",
    "customer and client data",
    "code consulting"
]

EXAMPLE_TEXTS = {
    "personal financial information": [
        "I just checked my bank account and noticed some charges I don't recognize. Could you help me figure out what's going on?",
        "I'm thinking about investing in some stocks, but I'm not sure where to start. Any advice on how to begin?"
    ],
    "company financial information": [
        "Our quarterly revenue has increased by 15%, and expenses have decreased. Let's prepare a report to share with stakeholders.",
        "We need to analyze the budget for the upcoming project to ensure we stay within our financial constraints."
    ],
    "human resources and employment": [
        "We're planning to hire a new marketing manager. Can you draft a job description and outline the interview process?",
        "An employee has requested information about their benefits package. Could you provide the necessary details?"
    ],
    "legal consulting": [
        "We received a notice about new data protection regulations. Let's review our policies to ensure compliance.",
        "There's a contract that needs to be reviewed before we proceed with the partnership. Can you take a look?"
    ],
    "health and medical information": [
        "I have a doctor's appointment next week and need to bring my medical records. How can I access them online?",
        "I've been feeling under the weather lately. Should I schedule a check-up or wait a few more days?"
    ],
    "customer and client data": [
        "A client has updated their contact information. Please make sure our records are current.",
        "We've received feedback from several customers about our service. Let's compile the comments and address any concerns."
    ],
    "code consulting": [
        "I'm trying to write a function that calculates the average of a list of numbers. Can you help me with a code example?",
        "I'm working on a project that requires me to write a Python script that will update its cached DNS records once an hour. Can you help me with the syntax?"
    ]
}

# Set page configuration
st.set_page_config(
    page_title="Cato GenAI Topic Analyzer",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all models as zero-shot classifiers."""
    loaded_models = {}
    for model_name, model_path in MODELS.items():
        classifier = pipeline("zero-shot-classification", 
                            model=model_path, 
                            device=0 if torch.cuda.is_available() else -1)
        loaded_models[model_name] = classifier
    return loaded_models

def predict(text, classifier):
    """Make a prediction using the zero-shot classifier."""
    output = classifier(text, 
                       candidate_labels=CATEGORIES,
                       multi_label=False)
    return output['scores']

def create_comparison_bar_chart(categories, all_probabilities, model_names):
    """Create a horizontal bar chart comparing multiple models."""
    fig = go.Figure()
    
    # Define a color palette for different models
    colors = ['rgb(55, 83, 109)', 'rgb(26, 118, 255)', 'rgb(0, 177, 106)']
    
    # Add bars for each model
    for idx, (model_name, probabilities) in enumerate(zip(model_names, all_probabilities)):
        fig.add_trace(go.Bar(
            name=model_name,
            y=categories,
            x=probabilities,
            orientation='h',
            text=[f'{p:.1%}' for p in probabilities],
            textposition='outside',
            textfont=dict(size=12),
            marker_color=colors[idx % len(colors)]
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Model Comparison - Category Probabilities',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title="Probability",
        yaxis_title="Category",
        xaxis=dict(
            tickformat='.0%',
            range=[0, 1.15]
        ),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

def display_results(text, loaded_models):
    """Display the analysis results for all models."""
    with st.spinner("Analyzing..."):
        all_probabilities = []
        model_predictions = {}
        
        # Get predictions from all models
        for model_name, classifier in loaded_models.items():
            probabilities = predict(text, classifier)
            all_probabilities.append(probabilities)
            model_predictions[model_name] = {
                'probabilities': probabilities,
                'most_likely': CATEGORIES[probabilities.index(max(probabilities))],
                'confidence': max(probabilities)
            }
        
        # Display results
        st.subheader("Analysis Results")
        
        # Create and display the comparison bar chart
        fig = create_comparison_bar_chart(CATEGORIES, all_probabilities, list(loaded_models.keys()))
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary table
        st.subheader("Model Predictions Summary")
        col1, col2 = st.columns(2)
        
        # Show individual model results
        for model_name, predictions in model_predictions.items():
            with col1:
                st.markdown(f"**{model_name}**")
                st.markdown(f"Most likely category: **{predictions['most_likely']}** ({predictions['confidence']:.1%})")
        
        # Show differences between models
        with col2:
            st.markdown("**Model Comparison**")
            models = list(model_predictions.keys())
            if len(models) >= 2:
                model1, model2 = models[0], models[1]
                pred1 = model_predictions[model1]
                pred2 = model_predictions[model2]
                
                # Compare categories
                if pred1['most_likely'] == pred2['most_likely']:
                    st.success(f"Both models agree on: **{pred1['most_likely']}**")
                else:
                    st.warning(f"Models disagree:\n- {model1}: **{pred1['most_likely']}**\n- {model2}: **{pred2['most_likely']}**")
                
                # Compare confidence
                conf_diff = abs(pred1['confidence'] - pred2['confidence'])
                st.markdown(f"Confidence difference: **{conf_diff:.1%}**")

# Title and description
st.title("📚 Cato GenAI Topic Analyzer")
st.markdown("### AI-Security-Squad")

st.markdown("""
This application classifies text into one of seven categories of sensitive information:
- **Personal Financial Information**: Personal banking, credit cards, investments
- **Company Financial Information**: Corporate finances, revenue, expenses
- **Human Resources and Employment**: Employee data, hiring, compensation
- **Legal Consulting**: Regulatory matters, contracts, policies
- **Health and Medical Information**: Medical records, health data
- **Customer and Client Data**: Client records, customer information
- **Code Consulting**: Programming questions, code examples, implementation help
""")

# Load all models
try:
    loaded_models = load_models()
    st.success(f"✅ Successfully loaded {len(loaded_models)} models")
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# Input selection
input_method = st.radio("Choose input method:", ["Enter custom text", "Select from examples"])

if input_method == "Enter custom text":
    # Custom text input with form
    with st.form("classification_form"):
        text_input = st.text_area("Enter your text:", height=200)
        submitted = st.form_submit_button("Analyze")
        
        if submitted and text_input:
            display_results(text_input, loaded_models)
        elif submitted:
            st.warning("Please enter some text to classify.")
else:
    # Example selection with immediate analysis
    col1, col2 = st.columns([1, 2])
    
    with col1:
        category = st.selectbox("Select category:", list(EXAMPLE_TEXTS.keys()))
        example_index = st.radio("Select example:", [f"Example {i+1}" for i in range(len(EXAMPLE_TEXTS[category]))], horizontal=True)
    
    # Get the selected example text
    text_input = EXAMPLE_TEXTS[category][int(example_index[-1])-1]
    
    with col2:
        st.text_area("Selected example:", value=text_input, height=200, disabled=True)
    
    # Analyze immediately when an example is selected
    display_results(text_input, loaded_models) 