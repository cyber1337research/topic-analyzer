import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import plotly.graph_objects as go

# Define available models
MODELS = {
    "mDeBERTa Multilingual NLI (2mil7)": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "mDeBERTa Multilingual NLI": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
}

# Define example texts
EXAMPLE_TEXTS = {
    "Personal Financial Information": [
        "I just checked my bank account and noticed some charges I don't recognize. Could you help me figure out what's going on?",
        "I'm thinking about investing in some stocks, but I'm not sure where to start. Any advice on how to begin?"
    ],
    "Company Financial Information": [
        "Our quarterly revenue has increased by 15%, and expenses have decreased. Let's prepare a report to share with stakeholders.",
        "We need to analyze the budget for the upcoming project to ensure we stay within our financial constraints."
    ],
    "Human Resources and Employment": [
        "We're planning to hire a new marketing manager. Can you draft a job description and outline the interview process?",
        "An employee has requested information about their benefits package. Could you provide the necessary details?"
    ],
    "Legal Consulting": [
        "We received a notice about new data protection regulations. Let's review our policies to ensure compliance.",
        "There's a contract that needs to be reviewed before we proceed with the partnership. Can you take a look?"
    ],
    "Health and Medical Information": [
        "I have a doctor's appointment next week and need to bring my medical records. How can I access them online?",
        "I've been feeling under the weather lately. Should I schedule a check-up or wait a few more days?"
    ],
    "Customer and Client Data": [
        "A client has updated their contact information. Please make sure our records are current.",
        "We've received feedback from several customers about our service. Let's compile the comments and address any concerns."
    ],
    "Code Consulting": [
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
def load_model(_model_name):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(_model_name)
    return model, tokenizer

def create_horizontal_bar_chart(categories, probabilities):
    """Create a horizontal bar chart with data labels using Plotly."""
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=categories,
        x=probabilities,
        orientation='h',
        text=[f'{p:.1%}' for p in probabilities],  # Format as percentage
        textposition='outside',
        textfont=dict(size=14),
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Category Probabilities',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title="Probability",
        yaxis_title="Category",
        xaxis=dict(
            tickformat='.0%',  # Format x-axis ticks as percentages
            range=[0, 1.15]    # Extend range slightly to accommodate labels
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

def predict(text, model, tokenizer):
    """Make a prediction using the model."""
    # For topic classification, we'll create hypothesis templates
    hypotheses = [
        "This text contains personal financial information.",
        "This text contains company financial information.",
        "This text contains human resources and employment information.",
        "This text contains legal consulting information.",
        "This text contains health and medical information.",
        "This text contains customer and client data.",
        "This text contains computer code consulting requests."
    ]
    
    # Store probabilities for each category
    probs = []
    
    # Process each hypothesis
    for hypothesis in hypotheses:
        inputs = tokenizer(
            text,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            # Get probability of "entailment" (index 0 in the model's output)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            entailment_prob = prediction[0][0].item()
            probs.append(entailment_prob)
            
    # Normalize probabilities to sum to 1
    total = sum(probs)
    normalized_probs = [p/total for p in probs]
    
    return normalized_probs

def display_results(text, model, tokenizer):
    """Display the analysis results for the given text."""
    with st.spinner("Analyzing..."):
        probabilities = predict(text, model, tokenizer)
        
        # Display results
        st.subheader("Analysis Results")
        
        # Create a more visual representation of the results
        categories = list(EXAMPLE_TEXTS.keys())
        
        # Create and display the horizontal bar chart
        fig = create_horizontal_bar_chart(categories, probabilities)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the most likely category
        most_likely = categories[probabilities.index(max(probabilities))]
        st.success(f"Most likely category: **{most_likely}** ({probabilities[probabilities.index(max(probabilities))]:.1%})")

# Title and description
st.title("📚 Cato GenAI Topic Analyzer")
st.markdown("### AI-Security-Squad")

# Model selection
selected_model_name = st.selectbox(
    "Select Model:",
    options=list(MODELS.keys()),
    help="Choose the model to use for analysis"
)

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

# Load model
try:
    model, tokenizer = load_model(MODELS[selected_model_name])
    st.success(f"✅ Model loaded successfully: {selected_model_name}")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Input selection
input_method = st.radio("Choose input method:", ["Enter custom text", "Select from examples"])

if input_method == "Enter custom text":
    # Custom text input with form
    with st.form("classification_form"):
        text_input = st.text_area("Enter your text:", height=200)
        submitted = st.form_submit_button("Analyze")
        
        if submitted and text_input:
            display_results(text_input, model, tokenizer)
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
    display_results(text_input, model, tokenizer) 