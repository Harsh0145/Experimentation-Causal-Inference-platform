from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import analyze_categorical, analyze_experiment, beta_binary_posterior, difference_in_differences, sample_size, sequential_analysis, synthetic_control

st.set_page_config(page_title="Fieldnotes · Experimentation Lab", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#152821;--muted:#71817a;--paper:#f5f6f2;--line:#e4e9e2;--green:#315e49;--mint:#e7f0e9;--amber:#f4c477}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)} .stApp{background:var(--paper)}
[data-testid="stSidebar"]{background:#182d24;color:#f6f5ef} [data-testid="stSidebar"] *{color:#eef4ee}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#aebdb2;font-size:.83rem}
.block-container{max-width:1440px;padding:2rem 3.1rem 4rem}.hero{padding:1.4rem 0 1.1rem;border-bottom:1px solid var(--line);margin-bottom:1.35rem}
.eyebrow,.mono{font-family:'DM Mono',monospace;letter-spacing:.1em;text-transform:uppercase;font-size:.7rem;color:var(--muted)}
.hero h1{font:700 2.3rem 'Manrope',sans-serif;letter-spacing:-.055em;margin:.3rem 0}.hero p{color:var(--muted);margin:0;max-width:660px}
.metric-card{background:white;border:1px solid var(--line);border-radius:12px;padding:16px 18px;min-height:104px}
.metric-label{font:500 .7rem 'DM Mono',monospace;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.metric-value{font:700 1.5rem 'Manrope',sans-serif;letter-spacing:-.04em;margin-top:7px}.metric-note{font-size:.75rem;color:var(--muted);margin-top:3px}
.insight{background:#e7f0e9;border-left:3px solid #54775e;border-radius:4px 10px 10px 4px;padding:15px 18px;line-height:1.55}
.stButton>button{border-radius:8px}.stTabs [data-baseweb="tab-list"]{gap:18px}.stTabs [data-baseweb="tab"]{padding:10px 4px}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden}
hr{border-color:var(--line)} footer{visibility:hidden}
</style>""", unsafe_allow_html=True)

@st.cache_data
def demo_data(seed: int = 32, n: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variant = rng.choice(["Control", "Treatment"], n)
    probability = np.where(variant == "Treatment", .124, .105)
    converted = rng.binomial(1, probability)
    selected_plan = np.array([rng.choice(["Basic", "Plus", "Premium"],
                                p=[.58, .30, .12] if arm == "Control" else [.44, .38, .18])
                              for arm in variant])
    return pd.DataFrame({"user_id": np.arange(100001, 100001+n), "variant": variant,
                         "converted": converted, "revenue": np.where(converted, rng.gamma(2.1, 21, n), 0),
                         "session_minutes": rng.gamma(2.5, 4.8, n) + (variant == "Treatment")*.35,
                         "selected_plan": selected_plan})

@st.cache_data
def causal_demo() -> pd.DataFrame:
    rng = np.random.default_rng(12); rows=[]
    for unit in ["City A", "City B", "City C", "City D", "City E"]:
        treated=unit == "City A"
        for t in range(20):
            level=100 + 4*t + rng.normal(0, 3)
            if treated and t >= 14: level += 17 + (t-14)*1.6
            rows.append({"unit":unit,"week":t,"outcome":level})
    return pd.DataFrame(rows)

st.sidebar.markdown("<div style='font:800 1.1rem Manrope;letter-spacing:-.04em;padding:10px 0'>◉ &nbsp;fieldnotes<span style='color:#a6c3a9'> / lab</span></div>", unsafe_allow_html=True)
st.sidebar.markdown("A workspace for evidence-led product decisions.")
st.sidebar.markdown("---")
st.sidebar.markdown("**WORKSPACE**")
page=st.sidebar.radio("Navigation",["Overview","Experiment analysis","Sequential testing","Power planner","Causal methods"],label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**DATA SOURCE**")
source=st.sidebar.radio("Dataset",["Illustrative experiment","Upload CSV"],label_visibility="collapsed")
uploaded=None
if source == "Upload CSV": uploaded=st.sidebar.file_uploader("Choose a CSV file",type=["csv"])
df=pd.read_csv(uploaded) if uploaded else demo_data()
st.sidebar.caption(f"{len(df):,} rows · {len(df.columns)} columns")
st.sidebar.markdown("---")
st.sidebar.caption("FIELDNOTES  /  EXPERIMENTATION STUDIO\n\nMethods are decision support. Pair results with product context and a pre-agreed analysis plan.")

st.markdown('<div class="hero"><div class="eyebrow">EXPERIMENTATION & CAUSAL INFERENCE &nbsp; / &nbsp; WORKSPACE</div><h1>Make decisions with evidence.</h1><p>Plan a study, read the result, and understand what the data can—and cannot—tell you.</p></div>',unsafe_allow_html=True)

def metric(label,value,note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

if page == "Overview":
    st.markdown("### Your experiment workspace")
    st.caption("A guided path from study design to a defensible conclusion.")
    c1,c2,c3,c4=st.columns(4)
    metric("STUDY DESIGN","01","Estimate the sample you need")
    metric("FREQUENTIST","02","Effect, uncertainty, and p-value")
    metric("BAYESIAN","03","Probability treatment is better")
    metric("CAUSAL","04","Explore observational designs")
    st.write("")
    left,right=st.columns([1.25,.75],gap="large")
    with left:
        st.markdown("#### Start with the data")
        st.markdown('<div class="insight"><b>Working example</b><br/>A simulated product experiment compares a new onboarding flow with the existing experience. Explore conversion, revenue, and session time—or upload a CSV from your own experiment.</div>',unsafe_allow_html=True)
        st.write("")
        preview=df.head(8)
        st.dataframe(preview,use_container_width=True,hide_index=True)
    with right:
        st.markdown("#### A careful workflow")
        for i,(title,body) in enumerate([("01  Plan","Choose a meaningful effect and estimate the sample size."),("02  Analyze","Select outcomes, compare arms, and adjust for multiple tests."),("03  Interpret","Read uncertainty alongside the effect size and study design.")]):
            st.markdown(f"**{title}**  \n{body}")
            if i<2: st.markdown("---")
        st.caption("Demo data is generated locally on each app session. Uploaded files are processed in memory.")

elif page == "Experiment analysis":
    st.markdown("### Experiment analysis")
    st.caption("Compare two randomized arms and keep the uncertainty visible.")
    cols=list(df.columns)
    with st.container(border=True):
        a1,a2,a3,a4=st.columns(4)
        group=a1.selectbox("Assignment column",cols,index=cols.index("variant") if "variant" in cols else 0)
        outcome=a2.selectbox("Outcome column",[c for c in cols if c!=group],index=([c for c in cols if c!=group].index("converted") if "converted" in cols else 0))
        selected_values=df[outcome].dropna()
        if not pd.api.types.is_numeric_dtype(selected_values):
            kind_default=2
        elif set(selected_values.unique()).issubset({0,1}):
            kind_default=0
        else:
            kind_default=1
        kind_key=f"analysis_kind_{group}_{outcome}_{df[outcome].dtype}"
        kind=a3.selectbox("Outcome type",["Binary conversion","Continuous","Categorical (chi-square)"],index=kind_default,key=kind_key,help="Choose chi-square for named or numeric-coded categories, such as plan type or support reason.")
        alpha=a4.selectbox("Significance level",[.01,.05,.10],index=1,format_func=lambda x:f"{x:.0%}")
    arms=sorted(df[group].dropna().astype(str).unique())
    if len(arms)<2: st.error("The assignment column needs at least two groups."); st.stop()
    control,treatment=st.columns(2)
    with control: control_arm=st.selectbox("Control arm",arms,index=0)
    with treatment: treatment_arm=st.selectbox("Treatment arm",[x for x in arms if x!=control_arm],index=0)
    try:
        if kind == "Categorical (chi-square)":
            result=analyze_categorical(df,group,outcome,control_arm,treatment_arm,alpha)
            c1,c2,c3,c4=st.columns(4)
            metric("CONTROL n",f"{result['n_control']:,}",control_arm)
            metric("TREATMENT n",f"{result['n_treatment']:,}",treatment_arm)
            metric("CRAMÉR’S V",f"{result['cramers_v']:.3f}",f"χ²({result['degrees_freedom']}) = {result['statistic']:.2f}")
            metric("P-VALUE",f"{result['p_value']:.4f}","Chi-square test of independence")
            st.write("")
            if result["significant"]:
                conclusion="The distribution of categories differs between these arms at the selected threshold."
            else:
                conclusion="There is not enough evidence at the selected threshold to say the category distributions differ."
            st.markdown(f'<div class="insight"><b>{conclusion}</b> Chi-square tests whether the distributions are associated; it does not identify which category caused the difference or establish causation on its own.</div>',unsafe_allow_html=True)
            if result["expected_min"] < 5:
                st.warning(f"At least one expected cell count is below 5 (minimum {result['expected_min']:.2f}). The chi-square approximation may be unreliable; combine rare categories when substantively appropriate or use an exact/Monte Carlo method.")
            table=result["contingency"]
            shares=table.div(table.sum(axis=1),axis=0).rename_axis(index=group,columns=outcome).reset_index().melt(id_vars=group,var_name=outcome,value_name="Share")
            fig=px.bar(shares,x=group,y="Share",color=outcome,barmode="stack",text_auto=".0%",color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(height=340,margin=dict(l=12,r=12,t=20,b=12),yaxis_title="Share within arm",xaxis_title="",yaxis_tickformat=".0%",plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
            with st.expander("Observed category counts"):
                st.dataframe(table,use_container_width=True)
        else:
            result=analyze_experiment(df,group,outcome,kind,control_arm,treatment_arm,alpha)
            value_format=".1%" if kind=="Binary conversion" else ".3f"
            lift=result["effect"]
            c1,c2,c3,c4=st.columns(4)
            metric("CONTROL MEAN",format(result["mean_a"],value_format),f"n = {result['n_a']:,}")
            metric("TREATMENT MEAN",format(result["mean_b"],value_format),f"n = {result['n_b']:,}")
            metric("ABSOLUTE EFFECT",format(lift,"+.2%" if kind=="Binary conversion" else "+.3f"),f"95% CI {result['ci_low']:+.3%} to {result['ci_high']:+.3%}" if kind=="Binary conversion" else f"95% CI {result['ci_low']:+.3f} to {result['ci_high']:+.3f}")
            metric("P-VALUE",f"{result['p_value']:.4f}",result["test"])
            st.write("")
            if kind == "Binary conversion":
                post=beta_binary_posterior(df.loc[df[group].astype(str)==control_arm,outcome].dropna().to_numpy(),df.loc[df[group].astype(str)==treatment_arm,outcome].dropna().to_numpy())
                st.markdown(f'<div class="insight"><b>Bayesian read:</b> given a uniform Beta(1,1) prior and this data, there is a <b>{post["prob_b_better"]:.1%}</b> probability that treatment conversion is higher. The posterior mean lift is {post["expected_lift"]:+.2%} (95% credible interval {post["ci_low"]:+.2%} to {post["ci_high"]:+.2%}).</div>',unsafe_allow_html=True)
            else:
                label="Evidence against no difference" if result["significant"] else "Inconclusive at the selected threshold"
                st.markdown(f'<div class="insight"><b>{label}.</b> The estimated change is {lift:+.4g}; the confidence interval is {result["ci_low"]:+.4g} to {result["ci_high"]:+.4g}. A p-value does not measure the probability that the hypothesis is true.</div>',unsafe_allow_html=True)
            st.write("")
            fig=go.Figure()
            fig.add_trace(go.Bar(x=[control_arm,treatment_arm],y=[result["mean_a"],result["mean_b"]],marker_color=["#b9c6bb","#315e49"],text=[f"{result['mean_a']:.3f}",f"{result['mean_b']:.3f}"],textposition="outside"))
            fig.update_layout(height=320,margin=dict(l=12,r=12,t=20,b=12),yaxis_title=outcome,xaxis_title="",plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    except (ValueError,TypeError) as exc: st.error(str(exc))
    with st.expander("Multiple testing: correct a family of outcomes"):
        candidates=[c for c in cols if c!=group]
        selected=st.multiselect("Outcomes to test together",candidates,default=[outcome])
        if selected:
            categorical_defaults=[outcome] if kind=="Categorical (chi-square)" and outcome in selected else []
            categorical_family=st.multiselect("Treat these outcomes as categories",selected,default=categorical_defaults,help="Useful for numeric coded categories such as 1 = basic, 2 = plus, 3 = premium.")
            rows=[]
            for col in selected:
                try:
                    non_missing=df[col].dropna()
                    if col in categorical_family or (not pd.api.types.is_numeric_dtype(non_missing) and not set(non_missing.unique()).issubset({0,1})):
                        p_value=analyze_categorical(df,group,col,control_arm,treatment_arm,alpha)["p_value"]
                    else:
                        outcome_kind="Binary conversion" if set(non_missing.unique()).issubset({0,1}) else "Continuous"
                        p_value=analyze_experiment(df,group,col,outcome_kind,control_arm,treatment_arm,alpha)["p_value"]
                    rows.append((col,p_value))
                except (ValueError,TypeError): pass
            if rows:
                from statsmodels.stats.multitest import multipletests
                adj=multipletests([x[1] for x in rows],alpha=alpha,method="bonferroni")[1]
                bh=multipletests([x[1] for x in rows],alpha=alpha,method="fdr_bh")[1]
                st.dataframe(pd.DataFrame({"Outcome":[x[0] for x in rows],"Raw p-value":[x[1] for x in rows],"Bonferroni p":[*adj],"BH adjusted p":[*bh]}).style.format(precision=4),hide_index=True,use_container_width=True)

elif page == "Sequential testing":
    st.markdown("### Sequential testing")
    st.caption("Review accumulating experiment results at pre-planned checkpoints, with a family-wise error bound across looks.")
    columns=list(df.columns)
    if len(columns)<2:
        st.error("Upload a CSV with an assignment column and numeric outcome column."); st.stop()
    with st.container(border=True):
        s1,s2,s3,s4=st.columns(4)
        group=s1.selectbox("Assignment column",columns,index=columns.index("variant") if "variant" in columns else 0,key="seq_group")
        outcomes=[c for c in columns if c!=group]
        outcome=s2.selectbox("Outcome column",outcomes,index=outcomes.index("converted") if "converted" in outcomes else 0,key="seq_outcome")
        kind=s3.selectbox("Outcome type",["Binary conversion","Continuous"],key="seq_kind")
        alpha=s4.selectbox("Family-wise alpha",[.01,.05,.10],index=1,format_func=lambda x:f"{x:.0%}",key="seq_alpha")
        s5,s6,s7=st.columns(3)
        order_choices=["CSV row order"]+[c for c in columns if c not in (group,outcome)]
        likely_order=next((c for c in order_choices if c.lower() in {"timestamp","event_time","created_at","date","time","user_id","id"}),"CSV row order")
        order_by=s5.selectbox("Arrival order",order_choices,index=order_choices.index(likely_order),help="Choose the timestamp, sequence, or ID that represents when observations arrived. CSV row order is used if there is no such column.")
        looks=s6.number_input("Planned looks",min_value=2,max_value=20,value=5,step=1)
    arms=sorted(df[group].dropna().astype(str).unique())
    if len(arms)<2:
        st.error("The assignment column needs at least two groups."); st.stop()
    a1,a2=st.columns(2)
    control=a1.selectbox("Control arm",arms,index=0,key="seq_control")
    treatment=a2.selectbox("Treatment arm",[x for x in arms if x!=control],index=0,key="seq_treatment")
    if order_by != "CSV row order":
        ordered=df.sort_values(order_by,kind="stable",na_position="last")
    else:
        ordered=df.copy()
    try:
        results=sequential_analysis(ordered,group,outcome,kind,control,treatment,int(looks),alpha)
        threshold=alpha/int(looks)
        crossed=results[results["Decision"]=="Crossed boundary"]
        latest=results.iloc[-1]
        m1,m2,m3=st.columns(3)
        metric("LOOKS PLANNED",str(int(looks)),"fixed before analysis")
        metric("PER-LOOK THRESHOLD",f"{threshold:.3%}",f"overall alpha ≤ {alpha:.0%}")
        if not crossed.empty:
            first=int(crossed.iloc[0]["Look"])
            metric("SEQUENTIAL DECISION","Boundary crossed",f"first crossed at look {first}")
        else:
            metric("SEQUENTIAL DECISION","No boundary crossed",f"latest adjusted p = {latest['Adjusted p-value']:.4g}" if pd.notna(latest["Adjusted p-value"]) else "Not enough observations yet")
        st.write("")
        st.markdown('<div class="insight"><b>How to use this safely:</b> pre-specify the maximum number of looks and examine results only at these checkpoints. This page divides alpha by the number of looks (Bonferroni), which controls the chance of any false positive at or below the chosen family-wise alpha, but is more conservative than a purpose-built group-sequential design. Do not change outcomes or inspect unscheduled checkpoints after seeing results.</div>',unsafe_allow_html=True)
        st.write("")
        table=results.copy()
        st.dataframe(table.style.format({"Raw p-value":"{:.4g}","Adjusted p-value":"{:.4g}","Effect":"{:+.4g}"},na_rep="—"),hide_index=True,use_container_width=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=results["Look"],y=results["Raw p-value"],mode="lines+markers",name="Raw p-value",line_color="#315e49"))
        fig.add_hline(y=threshold,line_dash="dash",line_color="#d28a3b",annotation_text=f"Per-look threshold {threshold:.3%}")
        fig.update_layout(height=330,margin=dict(l=10,r=10,t=22,b=10),xaxis_title="Planned look",yaxis_title="Raw p-value",yaxis_range=[0,1],plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    except (ValueError,TypeError) as exc:
        st.error(str(exc))

elif page == "Power planner":
    st.markdown("### Power planner")
    st.caption("Estimate the number of observations required in each arm for a two-sided test of means.")
    with st.container(border=True):
        x1,x2,x3,x4=st.columns(4)
        effect=x1.number_input("Standardized effect size (Cohen’s d)",min_value=.05,max_value=2.0,value=.25,step=.05,help="Difference in means divided by the outcome standard deviation.")
        power=x2.slider("Target power",.60,.99,.80,.01)
        alpha=x3.selectbox("Type I error rate",[.01,.05,.10],index=1,format_func=lambda x:f"{x:.0%}")
        ratio=x4.number_input("Treatment : control allocation",min_value=.25,max_value=4.0,value=1.0,step=.25)
    n=sample_size(effect,alpha,power,ratio)
    cols=st.columns(3); metric("CONTROL SAMPLE",f"{n:,}","participants required"); metric("TREATMENT SAMPLE",f"{int(np.ceil(n*ratio)):,}",f"allocation ratio {ratio:g}:1"); metric("TOTAL SAMPLE",f"{n+int(np.ceil(n*ratio)):,}",f"at {power:.0%} power and {alpha:.0%} alpha")
    st.write("")
    st.markdown('<div class="insight"><b>Planning note.</b> This approximation assumes independent observations, equal variance, a fixed sample, and a normally distributed continuous outcome. For binary outcomes, use baseline conversion and minimum detectable absolute lift to plan with a dedicated proportion model.</div>',unsafe_allow_html=True)
    st.markdown("#### How sample size changes with power")
    powers=np.linspace(.65,.95,16); sizes=[sample_size(effect,alpha,float(p),ratio) for p in powers]
    fig=px.line(x=powers,y=[v*(1+ratio) for v in sizes],markers=True,labels={"x":"Target power","y":"Total participants"})
    fig.update_traces(line_color="#315e49",marker_color="#315e49");fig.update_layout(height=330,margin=dict(l=8,r=8,t=20,b=5),plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)

else:
    st.markdown("### Causal methods")
    st.caption("Methods for settings without random assignment. A credible estimate depends on design assumptions.")
    method=st.radio("Method",["Difference-in-differences","Synthetic control"],horizontal=True)
    cdf=causal_demo() if source=="Illustrative experiment" else df
    if method=="Difference-in-differences":
        c1,c2,c3=st.columns(3)
        unit=c1.selectbox("Group / unit",cdf.columns,index=cdf.columns.get_loc("unit") if "unit" in cdf else 0)
        time=c2.selectbox("Period",cdf.columns,index=cdf.columns.get_loc("week") if "week" in cdf else min(1,len(cdf.columns)-1))
        outcome=c3.selectbox("Outcome",cdf.columns,index=cdf.columns.get_loc("outcome") if "outcome" in cdf else min(2,len(cdf.columns)-1))
        groups=sorted(cdf[unit].dropna().unique().tolist(),key=str); periods=sorted(cdf[time].dropna().unique().tolist())
        if len(groups)>=2 and len(periods)>=2:
            cutoff_index=min(max(int(len(periods)*.7),1),len(periods)-1)
            treat=st.selectbox("Treated group",groups); post=st.selectbox("Treatment begins",periods,index=cutoff_index)
            try:
                did=difference_in_differences(cdf,outcome,unit,time,treat,post)
                metric("DIFFERENCE-IN-DIFFERENCES",f"{did['effect']:+.3f}","change in outcome units")
                pvt=cdf.groupby([time,unit])[outcome].mean().reset_index()
                fig=px.line(pvt,x=time,y=outcome,color=unit,markers=True)
                fig.update_layout(height=350,margin=dict(l=8,r=8,t=10,b=8),plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)");st.plotly_chart(fig,use_container_width=True)
                st.markdown('<div class="insight"><b>Assumption to examine:</b> absent treatment, treated and comparison groups would have followed parallel outcome trends. This before/after contrast does not test that assumption; inspect pre-treatment trends and potential spillovers.</div>',unsafe_allow_html=True)
            except ValueError as e: st.warning(str(e))
        else: st.info("Difference-in-differences needs at least two groups and exactly two periods in this starter view.")
    else:
        c1,c2,c3=st.columns(3)
        unit=c1.selectbox("Unit",cdf.columns,index=cdf.columns.get_loc("unit") if "unit" in cdf else 0,key="sc_unit")
        time=c2.selectbox("Time",cdf.columns,index=cdf.columns.get_loc("week") if "week" in cdf else min(1,len(cdf.columns)-1),key="sc_time")
        outcome=c3.selectbox("Outcome",cdf.columns,index=cdf.columns.get_loc("outcome") if "outcome" in cdf else min(2,len(cdf.columns)-1),key="sc_outcome")
        units=sorted(cdf[unit].dropna().unique().tolist(),key=str); treated=st.selectbox("Treated unit",units)
        time_values=sorted(cdf[time].dropna().unique().tolist())
        default_intervention=min(max(int(len(time_values)*.7),1),len(time_values)-1) if len(time_values)>1 else 0
        intervention=st.selectbox("Intervention begins",time_values,index=default_intervention)
        try:
            sc=synthetic_control(cdf,unit,time,outcome,treated,intervention)
            fig=go.Figure();fig.add_scatter(x=sc.time,y=sc["Observed treated"],name="Treated",line_color="#315e49");fig.add_scatter(x=sc.time,y=sc["Synthetic control"],name="Synthetic",line=dict(color="#e09b45",dash="dash"));fig.add_vline(x=intervention,line_dash="dot",line_color="#829087")
            fig.update_layout(height=360,margin=dict(l=8,r=8,t=10,b=8),plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)");st.plotly_chart(fig,use_container_width=True)
            avg=sc.loc[sc.time>=intervention,"Estimated effect"].mean();metric("POST-PERIOD GAP",f"{avg:+.3f}","observed minus synthetic outcome")
            st.markdown('<div class="insight"><b>Interpret with care.</b> This demo builds a nonnegative, sum-to-one donor pool using pre-period outcome fit. Credibility depends on donor quality, pre-period fit, no spillovers, and no concurrent shocks affecting only the treated unit.</div>',unsafe_allow_html=True)
        except (ValueError,KeyError) as e: st.warning(str(e))

st.markdown("---")
st.caption("FIELDNOTES / LAB  ·  Statistical estimates are decision support, not a substitute for study design.")
