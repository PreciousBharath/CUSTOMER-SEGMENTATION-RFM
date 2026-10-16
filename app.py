"""SegmentIQ - Customer segmentation with RFM analysis & K-Means clustering.
Run:  streamlit run app.py
"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from segmentation import SEGMENT_ACTIONS, cluster, k_search, make_transactions, rfm_table

st.set_page_config(page_title="SegmentIQ", page_icon="🧩", layout="wide")
st.markdown("<h1 style='margin-bottom:0'>🧩 SegmentIQ — Customer Segmentation</h1>"
            "<p style='color:#888'>RFM feature engineering → log scaling → K-Means → actionable marketing segments</p>",
            unsafe_allow_html=True)
COL = {"Champions": "#39D98A", "Loyal Customers": "#00C9FF", "At Risk": "#FF416C", "Promising / New": "#F7B32B", "Hibernating": "#8E2DE2"}


@st.cache_data
def pipeline():
    tx = make_transactions()
    rfm = rfm_table(tx)
    return tx, rfm, k_search(__import__("segmentation").scale(rfm))


tx, rfm, ks = pipeline()
k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)
seg, sil = cluster(rfm, k)
cmap = COL

m = st.columns(5)
m[0].metric("Customers", f"{len(rfm):,}")
m[1].metric("Orders", f"{len(tx):,}")
m[2].metric("Revenue", f"₹{tx.amount.sum()/1e7:.2f} Cr")
m[3].metric("Clusters", k)
m[4].metric("Silhouette", f"{sil:.3f}")

c1, c2 = st.columns([3, 2])
with c1:
    fig = px.scatter_3d(seg.sample(2000, random_state=1), x="recency", y="frequency", z="monetary", color="segment",
                        color_discrete_map=cmap, opacity=.75, log_y=True, log_z=True, title="RFM space (3D)")
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=470, margin=dict(t=40, b=0, l=0, r=0), legend=dict(orientation="h", y=-0.02))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks.k, y=ks.inertia, name="Inertia (elbow)", line=dict(color="#F7B32B")))
    fig.add_trace(go.Scatter(x=ks.k, y=ks.silhouette, name="Silhouette", yaxis="y2", line=dict(color="#00C9FF")))
    fig.add_vline(x=k, line_dash="dash", line_color="#FF416C")
    fig.update_layout(title="Choosing k: elbow + silhouette", height=230, margin=dict(t=40, b=10),
                      yaxis2=dict(overlaying="y", side="right"), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    share = seg.groupby("segment").agg(customers=("segment", "size"), revenue=("monetary", "sum")).reset_index()
    share[["customers", "revenue"]] = 100 * share[["customers", "revenue"]] / share[["customers", "revenue"]].sum()
    fig = px.bar(share.melt(id_vars="segment"), x="value", y="variable", color="segment", orientation="h", color_discrete_map=cmap, title="Share of customers vs revenue (%)")
    fig.update_layout(height=230, margin=dict(t=40, b=10), showlegend=False, yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

st.subheader("📋 Segment profiles & recommended actions")
prof = seg.groupby("segment").agg(customers=("cluster", "size"), avg_recency_days=("recency", "mean"),
                                  avg_orders=("frequency", "mean"), avg_spend=("monetary", "mean"),
                                  total_revenue=("monetary", "sum")).round(0).sort_values("avg_spend", ascending=False)
prof["action"] = [SEGMENT_ACTIONS.get(s.rstrip(" 0123456789"), "—") for s in prof.index]
st.dataframe(prof, width="stretch",
             column_config={"avg_spend": st.column_config.NumberColumn(format="₹%d"),
                            "total_revenue": st.column_config.NumberColumn(format="₹%d")})

c1, c2 = st.columns(2)
with c1:
    fig = px.box(seg, x="segment", y="monetary", color="segment", color_discrete_map=cmap, log_y=True,
                 title="Spend distribution per segment")
    fig.update_layout(height=320, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.scatter(seg, x="pc1", y="pc2", color="segment", color_discrete_map=cmap, opacity=.6,
                     title="PCA projection of clusters")
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=320, margin=dict(t=40, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

st.download_button("⬇️ Download customer segments (CSV)", seg.reset_index().to_csv(index=False), "customer_segments.csv")
