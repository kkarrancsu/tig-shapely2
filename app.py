import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import itertools
import math
from typing import Dict, List, Set
from scipy.stats import beta, gaussian_kde

def v(coalition: Set[str], penalties: Dict[str, float], require_ai: bool) -> float:
    # characteristic function: value of a coalition
    S = set(coalition)
    if require_ai and 'AI' not in S:
        return 0.0
    # start from total potential value = 1.0
    val = 1.0
    # subtract penalty for each missing role
    for p, pen in penalties.items():
        if p not in S:
            val -= pen
    # ensure non-negative
    return max(val, 0.0)

def compute_shapley(penalties: Dict[str, float], require_ai: bool = True) -> Dict[str, float]:
    players = list(penalties.keys())
    n = len(players)

    # initialize Shapley values
    shapley = {p: 0.0 for p in players}
    factorial = math.factorial

    # iterate over each role
    for p in players:
        others = [q for q in players if q != p]
        # consider all subsets of the other players
        for k in range(len(others) + 1):
            for subset in itertools.combinations(others, k):
                S = set(subset)
                marginal = v(S.union({p}), penalties, require_ai) - v(S, penalties, require_ai)
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                shapley[p] += weight * marginal

    return shapley

def sample_penalties(penalty_means: Dict[str, float], alpha_beta_params: Dict[str, Dict[str, float]], n_samples: int) -> List[Dict[str, float]]:
    """Sample penalties from beta distributions based on alpha and beta parameters."""
    samples = []
    for _ in range(n_samples):
        sample = {}
        for role, mean in penalty_means.items():
            params = alpha_beta_params[role]
            if params['alpha'] == 0 and params['beta'] == 0:  # Point estimate
                sample[role] = mean
            else:
                sample[role] = beta.rvs(params['alpha'], params['beta'])
        samples.append(sample)
    return samples

def compute_shapley_distribution(penalty_means: Dict[str, float], alpha_beta_params: Dict[str, Dict[str, float]], 
                               require_ai: bool, n_samples: int = 1000) -> Dict[str, np.ndarray]:
    """Compute distribution of Shapley values by sampling from penalty distributions."""
    samples = sample_penalties(penalty_means, alpha_beta_params, n_samples)
    shapley_distributions = {role: [] for role in penalty_means.keys()}
    
    for sample in samples:
        shapley_vals = compute_shapley(sample, require_ai)
        for role, val in shapley_vals.items():
            shapley_distributions[role].append(val)
    
    return {role: np.array(vals) for role, vals in shapley_distributions.items()}

st.title("Shapley Value Calculator")

st.markdown(r"""
### Coalition Value Equation

The value of a coalition $S$ is computed as:

$$
v(S) = \begin{cases} 
0 & \text{if require\_AI = True and AI} \notin S \\
\max\left(1 - \sum_{p \notin S} \text{penalty}_p, 0\right) & \text{otherwise}
\end{cases}
$$

Where $\text{penalty}_p$ is the penalty for missing player $p$ from the coalition.
""")

st.markdown("---")

# Sidebar configuration
st.sidebar.header("Configuration")

require_ai = st.sidebar.checkbox("Require AI (Coalition without AI = 0)", value=True)

penalties = {}
penalties['AI'] = st.sidebar.slider(
    "AI Missing Penalty", 
    0.0, 1.0, 1.0, 0.01,
    disabled=require_ai,
    help="This penalty only applies when 'Require AI' is unchecked"
)
penalties['B'] = st.sidebar.slider("Benchmarkers (B) Missing Penalty", 0.0, 1.0, 0.50, 0.01)
penalties['CI'] = st.sidebar.slider("Code Innovators (CI) Missing Penalty", 0.0, 1.0, 0.1, 0.01)
penalties['CM'] = st.sidebar.slider("Challenge Maintainers (CM) Missing Penalty", 0.0, 1.0, 0.20, 0.01)

# Create tabs
tab1, tab2 = st.tabs(["Point Estimates", "Uncertainty Analysis"])

with tab1:
    st.title("Shapley Value Calculator - Point Estimates")
    
    st.markdown(r"""
    ### Coalition Value Equation

    The value of a coalition $S$ is computed as:

    $$
    v(S) = \begin{cases} 
    0 & \text{if require\_AI = True and AI} \notin S \\
    \max\left(1 - \sum_{p \notin S} \text{penalty}_p, 0\right) & \text{otherwise}
    \end{cases}
    $$

    Where $\text{penalty}_p$ is the penalty for missing player $p$ from the coalition.
    """)

    st.markdown("---")
    
    # Compute Shapley values
    shapley_vals = compute_shapley(penalties, require_ai)

    # Create bar chart using Altair
    data = pd.DataFrame({
        'Role': list(shapley_vals.keys()),
        'Shapley Value': list(shapley_vals.values())
    }).sort_values('Shapley Value', ascending=False)  # Sort by Shapley Value in descending order

    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Role:N', sort=None, title='Role'),
        y=alt.Y('Shapley Value:Q', title='Shapley Value'),
        color=alt.Color('Role:N', legend=None),
        tooltip=['Role', 'Shapley Value']
    ).properties(
        width=600,
        height=400,
        title='Shapley Value Allocation'
    )

    st.altair_chart(chart, use_container_width=True)

    # Show sum to verify
    total = sum(shapley_vals.values())
    st.write(f"Sum of Shapley values: {total:.4f}")

    # Show all coalitions and their values for reference
    st.subheader("All Coalition Values")
    players = list(penalties.keys())
    coalitions = []

    for k in range(1, len(players) + 1):
        for subset in itertools.combinations(players, k):
            coalition_set = set(subset)
            coalitions.append({
                'Coalition': ', '.join(subset),
                'Value': v(coalition_set, penalties, require_ai)
            })

    coalition_df = pd.DataFrame(coalitions).sort_values('Value', ascending=False)  # Sort by Value in descending order
    st.table(coalition_df.style.hide(axis="index"))

with tab2:
    st.title("Shapley Value Calculator - Uncertainty Analysis")
    
    st.markdown("""
    This tab shows the uncertainty in Shapley values by sampling from beta distributions for each penalty.
    The beta distributions are parameterized by the mean (from the point estimates) and variance (configurable below).
    """)
    
    # Sidebar configuration for uncertainty analysis
    st.sidebar.header("Uncertainty Configuration")
    n_samples = st.sidebar.slider("Number of Monte Carlo Samples", 100, 10000, 1000, 100)
    
    # Get penalty means from the main configuration
    penalty_means = penalties.copy()
    
    # Add beta distribution parameter configuration
    alpha_beta_params = {}
    st.sidebar.subheader("Beta Distribution Parameters")
    for role in penalties.keys():
        st.sidebar.markdown(f"**{role} Parameters**")
        col1, col2 = st.sidebar.columns(2)
        if role == 'CM': 
            alpha_start = 5.0
            beta_start = 20.0
        elif role == 'CI':
            alpha_start = 1.0
            beta_start = 10.0
        else:  # B
            alpha_start = 6.0
            beta_start = 6.0
        with col1:
            alpha = st.number_input(
                f"{role} Alpha",
                min_value=0.0,
                max_value=100.0,
                value=alpha_start,
                step=0.1,
                key=f"{role}_alpha",
                disabled=(role == 'AI' and require_ai),
                help="Alpha parameter of beta distribution. Set both alpha and beta to 0 for a point estimate."
            )
        with col2:
            beta_param = st.number_input(
                f"{role} Beta",
                min_value=0.0,
                max_value=100.0,
                value=beta_start,
                step=0.1,
                key=f"{role}_beta",
                disabled=(role == 'AI' and require_ai),
                help="Beta parameter of beta distribution. Set both alpha and beta to 0 for a point estimate."
            )
        alpha_beta_params[role] = {'alpha': alpha, 'beta': beta_param}
    
    # Compute Shapley value distributions
    shapley_dists = compute_shapley_distribution(penalty_means, alpha_beta_params, require_ai, n_samples)
    
    # Create prior distribution plots
    st.subheader("Prior Distributions")
    prior_data = []
    for role, mean in penalty_means.items():
        if role == 'AI' and require_ai:
            continue
        params = alpha_beta_params[role]
        if params['alpha'] == 0 and params['beta'] == 0:
            # Point estimate
            x = np.array([mean])
            y = np.array([1.0])
        else:
            # Generate points for the PDF
            x = np.linspace(0.001, 0.999, 200)  # Avoid 0 and 1 to prevent numerical issues
            y = beta.pdf(x, params['alpha'], params['beta'])
            
            # Handle any remaining numerical issues
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.clip(y, 0, 100)  # Cap very large values
        
        prior_data.extend([{'Role': role, 'x': xi, 'y': yi} for xi, yi in zip(x, y)])
    
    prior_df = pd.DataFrame(prior_data)
    prior_chart = alt.Chart(prior_df).mark_line().encode(
        x=alt.X('x:Q', title='Penalty Value', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('y:Q', title='Density'),
        color='Role:N',
        tooltip=['Role', 'x', 'y']
    ).properties(
        width=600,
        height=400,
        title='Prior Distributions for Penalties'
    )
    st.altair_chart(prior_chart, use_container_width=True)
    
    # Create Shapley value distribution plots
    st.subheader("Shapley Value Distributions")
    dist_data = []
    for role, dist in shapley_dists.items():
        # Compute KDE
        kde = gaussian_kde(dist)
        # Generate points for the PDF
        x = np.linspace(min(dist), max(dist), 200)
        y = kde(x)
        
        # Normalize the density to make it more visually comparable
        y = y / np.trapz(y, x)
        
        for val, density in zip(x, y):
            dist_data.append({
                'Role': role,
                'Value': val,
                'Density': density
            })
    
    dist_df = pd.DataFrame(dist_data)
    dist_chart = alt.Chart(dist_df).mark_line().encode(
        x=alt.X('Value:Q', title='Shapley Value'),
        y=alt.Y('Density:Q', title='Density'),
        color='Role:N',
        tooltip=['Role', 'Value', 'Density']
    ).properties(
        width=600,
        height=400,
        title='Distribution of Shapley Values'
    )
    
    # Calculate medians and other percentiles for each role
    percentiles = {}
    for role, dist in shapley_dists.items():
        percentiles[role] = {
            '25th': np.percentile(dist, 25),
            '50th': np.percentile(dist, 50),  # median
            '75th': np.percentile(dist, 75)
        }
    
    # Calculate total for each percentile
    totals = {
        '25th': sum(p['25th'] for p in percentiles.values()),
        '50th': sum(p['50th'] for p in percentiles.values()),
        '75th': sum(p['75th'] for p in percentiles.values())
    }
    
    # Calculate reward distributions
    reward_distributions = {}
    for role in percentiles.keys():
        reward_distributions[role] = {
            '25th': percentiles[role]['25th'] / totals['25th'],
            '50th': percentiles[role]['50th'] / totals['50th'],
            '75th': percentiles[role]['75th'] / totals['75th']
        }
    
    # Create median markers
    median_data = []
    for role, pcts in percentiles.items():
        median_data.append({
            'Role': role,
            'Median': pcts['50th'],
            'Density': 0  # Will be set to max density in the plot
        })
    
    # Add median markers to the chart
    median_df = pd.DataFrame(median_data)
    median_markers = alt.Chart(median_df).mark_rule(
        strokeDash=[5, 5]  # Dotted line
    ).encode(
        x='Median:Q',
        color='Role:N',
        tooltip=['Role', 'Median']
    )
    
    # Combine the plots
    final_chart = (dist_chart + median_markers).properties(
        width=600,
        height=400,
        title='Distribution of Shapley Values with Median Markers'
    )
    
    st.altair_chart(final_chart, use_container_width=True)
    
    # Show reward distribution table
    st.subheader("Reward Distribution (Based on Percentiles)")
    reward_data = []
    for role in reward_distributions.keys():
        reward_data.append({
            'Role': role,
            '25th Percentile': reward_distributions[role]['25th'],
            'Median (50th)': reward_distributions[role]['50th'],
            '75th Percentile': reward_distributions[role]['75th']
        })
    
    reward_df = pd.DataFrame(reward_data).sort_values('Median (50th)', ascending=False)
    
    st.table(reward_df.style.format({
        '25th Percentile': '{:.2%}',
        'Median (50th)': '{:.2%}',
        '75th Percentile': '{:.2%}'
    }))