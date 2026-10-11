import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Churn Dashboard", layout="wide")

st.title("Bank Customer Churn Dashboard")
st.write("Python, Pandas and Machine Learning project")

df = pd.read_csv("bank_customers.csv")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Churned Customers", int(df["churn"].sum()))
col3.metric("Churn Rate", f'{df["churn"].mean() * 100:.2f}%')

st.subheader("Customer Data")
st.dataframe(df, use_container_width=True)

status = st.selectbox(
    "Filter customers",
    ["All", "Churned", "Retained"]
)

filtered = df.copy()

if status == "Churned":
    filtered = df[df["churn"] == 1]
elif status == "Retained":
    filtered = df[df["churn"] == 0]

st.subheader("Filtered Customers")
st.dataframe(filtered, use_container_width=True)

st.subheader("Churn Distribution")
counts = df["churn"].value_counts()
counts.index = ["Retained" if x == 0 else "Churned" for x in counts.index]
st.bar_chart(counts)

st.subheader("Balance by Churn Status")
fig, ax = plt.subplots()
df.boxplot(column="balance", by="churn", ax=ax)
ax.set_title("Customer Balance")
ax.set_xlabel("Churn (0 = Retained, 1 = Churned)")
ax.set_ylabel("Balance")
fig.suptitle("")
st.pyplot(fig)

st.caption("Demo data is synthetic and not real bank customer data.")