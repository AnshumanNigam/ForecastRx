import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
df = pd.read_csv("simulated_pharmacy_data.csv", parse_dates=["date", "expiry_date"])
medicines = df["medicine_name"].unique()

st.title("💊 Medicine Demand Forecast & Reorder Recommendation")

selected_medicine = st.selectbox("Select a Medicine", medicines)

med_df = df[df["medicine_name"] == selected_medicine]

monthly_sales = med_df.groupby(df["date"].dt.to_period("M"))["quantity_sold"].sum().reset_index()
monthly_sales["date"] = monthly_sales["date"].dt.to_timestamp()
prophet_df = monthly_sales.rename(columns={"date": "ds", "quantity_sold": "y"})

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=6, freq="M")
forecast = model.predict(future)

latest_stock = med_df.sort_values("date").iloc[-1]["stock_on_hand"]
future_forecast = forecast[forecast["ds"] > df["date"].max()][["ds", "yhat"]].copy()
future_forecast["predicted_demand"] = future_forecast["yhat"].round().astype(int)
future_forecast["available_stock"] = latest_stock

stock = latest_stock
reorder_plan = []
stock_left = []

for _, row in future_forecast.iterrows():
    demand = row["predicted_demand"]
    shortage = max(demand - stock, 0)
    reorder = shortage
    stock = max(stock - demand, 0) + reorder
    reorder_plan.append(reorder)
    stock_left.append(stock)

future_forecast["reorder_quantity"] = reorder_plan
future_forecast["remaining_stock_after"] = stock_left
future_forecast["shortage_risk"] = future_forecast["reorder_quantity"] > 0

st.subheader(f"📊 Forecast & Reorder Plan for {selected_medicine}")
st.dataframe(
    future_forecast[["ds", "predicted_demand", "available_stock", "reorder_quantity", "remaining_stock_after", "shortage_risk"]]
    .rename(columns={
        "ds": "Month",
        "predicted_demand": "Predicted Demand",
        "available_stock": "Starting Stock",
        "reorder_quantity": "Reorder Qty",
        "remaining_stock_after": "Ending Stock",
        "shortage_risk": "Shortage Risk?"
    })
)

st.subheader("📦 Reorder Plan Chart")
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(future_forecast["ds"].dt.strftime('%b-%Y'), future_forecast["reorder_quantity"], color="orange")
ax.set_title(f"Recommended Reorder per Month: {selected_medicine}")
ax.set_ylabel("Reorder Units")
ax.set_xlabel("Month")
ax.grid(True)
st.pyplot(fig)
st.subheader("⬇️ Download Reorder Plan as CSV")
export_df = future_forecast[["ds", "predicted_demand", "available_stock", "reorder_quantity", "remaining_stock_after", "shortage_risk"]]
export_df.columns = ["Month", "Predicted Demand", "Starting Stock", "Reorder Qty", "Ending Stock", "Shortage Risk?"]

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"reorder_plan_{selected_medicine.lower()}.csv",
    mime="text/csv"
)
