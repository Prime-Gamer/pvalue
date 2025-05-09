import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import matplotlib.patches as patches

st.set_page_config(page_title="Understanding P-Values with Coin Tosses", layout="wide")

st.title("Understanding P-Values Through Coin Tosses")
st.markdown("""
This interactive app will help you understand what a p-value is, how it's calculated, 
and how to interpret it in the context of coin toss experiments.
""")

# Create sidebar for inputs
with st.sidebar:
    st.header("Simulation Parameters")
    n_tosses = st.slider("Number of tosses", min_value=10, max_value=1000, value=100, step=10)
    true_prob = st.slider("Expected probability of heads (null hypothesis)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    observed_heads = st.slider("Observed number of heads", min_value=0, max_value=n_tosses, value=53)
    alpha = st.slider("Significance level (α)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    
    test_type = st.radio(
        "Hypothesis Test Type:",
        ["Two-tailed (≠)", "Right-tailed (>)", "Left-tailed (<)"],
        index=0
    )

# Calculate p-value based on test type
def calculate_p_value(observed, n, p, test_type):
    if test_type == "Two-tailed (≠)":
        # Two-tailed p-value
        if observed <= n * p:
            p_value = 2 * stats.binom.cdf(observed, n, p)
        else:
            p_value = 2 * (1 - stats.binom.cdf(observed - 1, n, p))
        # Ensure p-value doesn't exceed 1.0
        p_value = min(p_value, 1.0)
    elif test_type == "Right-tailed (>)":
        # Right-tailed p-value (probability of observing >= observed heads)
        p_value = 1 - stats.binom.cdf(observed - 1, n, p)
    else:  # Left-tailed
        # Left-tailed p-value (probability of observing <= observed heads)
        p_value = stats.binom.cdf(observed, n, p)
    
    return p_value

p_value = calculate_p_value(observed_heads, n_tosses, true_prob, test_type)

# Main content
st.header("What is a P-value?")
st.markdown("""
A **p-value** is the probability of observing a result at least as extreme as what we actually observed, 
assuming that the null hypothesis is true. In our coin toss example:

- **Null Hypothesis (H₀)**: The coin has a probability of heads = {0}
- **Alternative Hypothesis (H₁)**: 
  - Two-tailed test: The coin's probability is not equal to {0}
  - Right-tailed test: The coin's probability is greater than {0}
  - Left-tailed test: The coin's probability is less than {0}
""".format(true_prob))

# P-value visualization
st.header("Visualizing the P-value")

# Create tabs for different visualizations
tab1, tab2 = st.tabs(["P-value Visualization", "Interactive Experiments"])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Calculate binomial probabilities
        x_values = np.arange(0, n_tosses + 1)
        pmf_values = stats.binom.pmf(x_values, n_tosses, true_prob)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the binomial PMF
        bars = ax.bar(x_values, pmf_values, alpha=0.7, color='steelblue', width=0.8)
        
        # Highlight the observed value
        observed_prob = stats.binom.pmf(observed_heads, n_tosses, true_prob)
        
        # Color the bars based on test type to visualize p-value region
        if test_type == "Two-tailed (≠)":
            # Find values that are as or more extreme than observed
            expected = n_tosses * true_prob
            for i in range(len(bars)):
                if (i <= observed_heads and observed_heads <= expected) or \
                   (i >= observed_heads and observed_heads >= expected) or \
                   (abs(i - expected) >= abs(observed_heads - expected)):
                    bars[i].set_color('red')
                    bars[i].set_alpha(0.7)
        
        elif test_type == "Right-tailed (>)":
            # Color bars for values >= observed
            for i in range(observed_heads, len(bars)):
                bars[i].set_color('red')
                bars[i].set_alpha(0.7)
                
        else:  # Left-tailed
            # Color bars for values <= observed
            for i in range(observed_heads + 1):
                bars[i].set_color('red')
                bars[i].set_alpha(0.7)
        
        # Highlight the observed value specifically
        bars[observed_heads].set_color('darkred')
        bars[observed_heads].set_alpha(1.0)
        
        # Add vertical line at expected value
        expected = n_tosses * true_prob
        ax.axvline(expected, color='black', linestyle='--', alpha=0.7, 
                  label=f"Expected: {expected:.1f}")
        
        ax.set_xlabel("Number of heads")
        ax.set_ylabel("Probability")
        ax.set_title(f"P-value Visualization: {p_value:.4f}\n" + 
                     f"(Red area = {p_value:.4f})")
        
        # Make the plot more readable
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **This visualization shows:**
        - The blue bars show the probability of each possible outcome under the null hypothesis
        - The red area represents the p-value - the probability of outcomes at least as extreme as what we observed
        - The vertical dashed line shows the expected value under the null hypothesis
        """)
    
    with col2:
        st.subheader("Interpreting Your Results")
        
        # Reject or fail to reject
        reject_null = p_value < alpha
        test_result = "Reject" if reject_null else "Fail to reject"
        
        result_color = "red" if reject_null else "green"
        
        st.markdown(f"""
        #### P-value: **<span style='color:{result_color}'>{p_value:.4f}</span>**
        #### Significance level (α): {alpha}
        
        **Decision: <span style='color:{result_color}'>{test_result} the null hypothesis</span>**
        
        #### What This Means:
        """, unsafe_allow_html=True)
        
        if reject_null:
            st.markdown("""
            The p-value is less than your chosen significance level (α). This suggests that 
            the observed result would be unlikely if the null hypothesis were true.
            
            In other words, there is statistically significant evidence to suggest that 
            the coin is not behaving as expected under the null hypothesis.
            """)
        else:
            st.markdown("""
            The p-value is greater than your chosen significance level (α). This means that 
            the observed result is reasonably likely to occur if the null hypothesis is true.
            
            In other words, there is not enough statistical evidence to conclude that 
            the coin is behaving differently than expected under the null hypothesis.
            """)
        
        # Display exact probabilities
        exact_prob = stats.binom.pmf(observed_heads, n_tosses, true_prob)
        st.markdown(f"""
        #### Exact probability of {observed_heads} heads in {n_tosses} tosses: 
        {exact_prob:.6f}
        
        #### Remember:
        - P-value is NOT the probability that the null hypothesis is true
        - P-value is the probability of observing your result (or more extreme) IF the null hypothesis is true
        - Small p-values suggest evidence against the null hypothesis
        """)

with tab2:
    st.subheader("Interactive P-value Experiment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        Let's run simulations to see how p-values behave under different scenarios.
        
        **Experiment:** Flip a coin multiple times and calculate the p-value for each experiment.
        """)
        
        num_experiments = st.slider("Number of experiments", 100, 10000, 1000, 100)
        sim_prob = st.slider("Actual probability of heads in simulation", 0.0, 1.0, true_prob, 0.05)
        
        if st.button("Run Experiments"):
            # Run simulations
            results = np.random.binomial(n_tosses, sim_prob, size=num_experiments)
            
            # Calculate p-values for each experiment
            p_values = []
            for result in results:
                p = calculate_p_value(result, n_tosses, true_prob, test_type)
                p_values.append(p)
            
            # Count how many resulted in rejecting H0
            rejections = sum(p <= alpha for p in p_values)
            rejection_rate = rejections / num_experiments
                
            # Store results in session state for the other column to use
            st.session_state['p_values'] = p_values
            st.session_state['rejection_rate'] = rejection_rate
            st.session_state['results'] = results
            
            st.markdown(f"""
            #### Results:
            - **Rejection rate**: {rejection_rate:.4f} ({rejections} out of {num_experiments})
            - **Expected rejection rate if H₀ is true**: {alpha}
            """)
            
            if abs(sim_prob - true_prob) < 0.01:
                st.success("The rejection rate is close to the significance level, as expected when the null hypothesis is true!")
            else:
                st.info(f"The rejection rate differs from α because the simulated coin (p={sim_prob}) differs from the null hypothesis (p={true_prob}).")
    
    with col2:
        if 'p_values' in st.session_state:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot histogram of p-values
            ax.hist(st.session_state['p_values'], bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(alpha, color='red', linestyle='--', label=f'α = {alpha}')
            
            ax.set_xlabel('P-value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of P-values from Experiments')
            ax.legend()
            
            st.pyplot(fig)
            
            # Plot outcomes distribution
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            
            ax2.hist(st.session_state['results'], bins=min(30, n_tosses//3), alpha=0.7, 
                    color='green', edgecolor='black', density=True)
            
            # Add theoretical distribution
            x = np.arange(0, n_tosses + 1)
            sim_pmf = stats.binom.pmf(x, n_tosses, sim_prob)
            null_pmf = stats.binom.pmf(x, n_tosses, true_prob)
            
            ax2.plot(x, sim_pmf, 'r-', lw=2, label=f'Simulated (p={sim_prob})')
            ax2.plot(x, null_pmf, 'k--', lw=2, label=f'Null Hypothesis (p={true_prob})')
            
            ax2.set_xlabel('Number of Heads')
            ax2.set_ylabel('Probability Density')
            ax2.set_title('Distribution of Experimental Results')
            ax2.legend()
            
            st.pyplot(fig2)
            
            st.markdown("""
            **Key insights:**
            - If the null hypothesis is true, p-values should be uniformly distributed
            - If the null hypothesis is false, p-values tend to be smaller
            - The rejection rate should be approximately equal to α when the null hypothesis is true
            """)

# Detailed explanation of p-values
st.header("P-value: A Deeper Understanding")

with st.expander("P-value Step-by-Step Calculation"):
    st.markdown(f"""
    ### How is the P-value calculated for a coin toss experiment?
    
    For our example with {observed_heads} heads in {n_tosses} tosses (expecting p = {true_prob}):
    
    #### Step 1: Define "as extreme or more extreme"
    This depends on the type of hypothesis test:
    - **Two-tailed test**: Results that deviate from the expected value at least as much as our observed value
    - **Right-tailed test**: Results with at least as many heads as observed
    - **Left-tailed test**: Results with at most as many heads as observed
    
    #### Step 2: Calculate the probability of these extreme outcomes
    For our test type ({test_type}), the calculation is:
    """)
    
    if test_type == "Two-tailed (≠)":
        expected = n_tosses * true_prob
        if observed_heads <= expected:
            st.latex(r"p\text{-value} = 2 \times P(X \leq " + str(observed_heads) + r")")
            st.markdown(f"Since our observed value ({observed_heads}) is below the expected value ({expected:.1f}), we calculate the probability of getting {observed_heads} or fewer heads, then multiply by 2 to account for both tails.")
        else:
            st.latex(r"p\text{-value} = 2 \times P(X \geq " + str(observed_heads) + r")")
            st.markdown(f"Since our observed value ({observed_heads}) is above the expected value ({expected:.1f}), we calculate the probability of getting {observed_heads} or more heads, then multiply by 2 to account for both tails.")
    
    elif test_type == "Right-tailed (>)":
        st.latex(r"p\text{-value} = P(X \geq " + str(observed_heads) + r")")
        st.markdown(f"We calculate the probability of getting {observed_heads} or more heads.")
    
    else:  # Left-tailed
        st.latex(r"p\text{-value} = P(X \leq " + str(observed_heads) + r")")
        st.markdown(f"We calculate the probability of getting {observed_heads} or fewer heads.")
    
    st.markdown(f"""
    #### Step 3: Compare to significance level
    - Our calculated p-value: {p_value:.4f}
    - Our chosen significance level (α): {alpha}
    
    Since p-value {'<' if p_value < alpha else '≥'} α, we {'' if p_value < alpha else 'fail to'} reject the null hypothesis.
    """)

with st.expander("Common Misconceptions About P-values"):
    st.markdown("""
    ### Common Misconceptions About P-values
    
    #### Misconception 1: P-value is the probability that the null hypothesis is true
    **Reality**: The p-value is the probability of observing a result at least as extreme as what we observed, assuming the null hypothesis is true.
    
    #### Misconception 2: A large p-value proves the null hypothesis
    **Reality**: A large p-value only indicates that we failed to find evidence against the null hypothesis, not that the null hypothesis is true.
    
    #### Misconception 3: P-value is the probability of making an error
    **Reality**: The p-value is not the probability of making a mistake. The significance level (α) is the probability of rejecting a true null hypothesis (Type I error).
    
    #### Misconception 4: A small p-value means the result is practically significant
    **Reality**: A small p-value only indicates statistical significance, not practical importance. The effect size matters too.
    
    #### Misconception 5: P = 0.05 is a magic threshold
    **Reality**: The significance level of 0.05 is arbitrary. The choice of threshold should depend on the context and costs of different types of errors.
    """)

with st.expander("P-value in Scientific Research"):
    st.markdown("""
    ### The Role of P-values in Scientific Research
    
    #### The Replication Crisis
    In recent years, scientists have become increasingly concerned about the "replication crisis" - the finding that many scientific studies with statistically significant results (p < 0.05) cannot be replicated. This has led to a reassessment of how p-values are used.
    
    #### P-hacking
    "P-hacking" refers to various practices researchers might use to obtain statistically significant results:
    - Running multiple analyses and only reporting those with p < 0.05
    - Adding or removing outliers to get p < 0.05
    - Stopping data collection when reaching p < 0.05
    
    #### Better Practices
    - **Pre-registration**: Specifying hypotheses and analyses before collecting data
    - **Effect sizes**: Reporting the magnitude of effects, not just p-values
    - **Confidence intervals**: Providing ranges of plausible values
    - **Multiple comparisons correction**: Adjusting p-values when testing multiple hypotheses
    - **Replication**: Conducting follow-up studies to verify findings
    """)

st.markdown("""
---
### How to Use This App to Learn About P-values

1. **Experiment with parameters** in the sidebar to see how they affect the p-value
2. **Compare different test types** (two-tailed, right-tailed, left-tailed)
3. **Run multiple experiments** to understand the distribution of p-values
4. **Read the explanations** to deepen your understanding

Remember: The p-value answers the question, "If the null hypothesis were true, how likely would we be to observe a result at least as extreme as what we actually observed?"
""")