# ---------------------- Imports ----------------------
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="Linear Regression Validator", page_icon="📈")
st.title("📊 Linear Regression Assumptions Checker")
st.caption("Upload a dataset and assess the key assumptions of a linear regression model.")

# ---------------------- File Upload ----------------------
with st.expander("📁 Upload Data", expanded=True):
    spectra = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if spectra:
        try:
            if spectra.name.endswith(".xlsx"):
                df = pd.read_excel(spectra)
            elif spectra.name.endswith(".csv"):
                df = pd.read_csv(spectra)
            else:
                st.error("Unsupported file format. Please upload a .csv or .xlsx file.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.subheader("Preview of uploaded data")
        st.dataframe(df.head(), use_container_width=True)

        # ---------------------- Column Selection ----------------------
        dep_col = st.selectbox("🎯 Select Dependent Variable (Target)", df.columns)

        # prepare indep candidate list (all columns except chosen dep)
        indep_cols_all = list(df.columns.drop(dep_col))
        indep_cols = st.multiselect("📌 Select Independent Variables (choose one or more)", indep_cols_all, default=indep_cols_all)

        if not indep_cols:
            st.warning("Select at least one independent variable to proceed.")
        else:
            # Allow removing further unwanted variables (optional)
            remove_cols = st.multiselect("❌ Optionally remove unwanted independent variables", indep_cols)
            # compute the chosen set after removals
            final_indep = [c for c in indep_cols if c not in remove_cols]

            st.markdown("**Selected variables summary:**")
            st.write(f"- Dependent (target): **{dep_col}**")
            st.write(f"- Independent variables (final): **{final_indep}**")

            # Ask about categoricals to encode
            possible_cat = [c for c in final_indep if df[c].dtype == 'object' or df[c].dtype.name == 'category' or df[c].nunique() < 20]
            cat_cols = st.multiselect("🔤 Select Categorical Variables to Encode (optional)", possible_cat)

            st.info("When you're happy with the target and independent variables, click **Confirm & Run Analysis** below. Model fitting and all assumption checks will run only after you confirm.")
            confirm = st.button("✅ Confirm & Run Analysis", type="primary")

            # Only after user presses confirm do we fit the model and run validations
            if confirm:
                # Basic input checks
                if len(final_indep) == 0:
                    st.error("No independent variables selected after removals. Please select at least one predictor.")
                else:
                    X = df[final_indep].copy()
                    y = df[dep_col].copy()

                    # Encode selected categorical variables
                    if cat_cols:
                        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

                    # convert bools to int
                    for col in X.columns:
                        if X[col].dtype == 'bool':
                            X[col] = X[col].astype(int)

                    # Ensure y is numeric
                    if not np.issubdtype(y.dtype, np.number):
                        try:
                            y = pd.to_numeric(y)
                            st.warning("Target column converted to numeric.")
                        except Exception as e:
                            st.error(f"Target variable is not numeric and couldn't be converted: {e}")
                            st.stop()

                    # Ensure there are numeric predictors for VIF, etc.
                    X_numeric = X.select_dtypes(include='number')
                    if X_numeric.shape[1] == 0:
                        st.error("No numeric predictors available after encoding. Cannot fit a linear regression.")
                        st.stop()

                    # Fit model
                    try:
                        model = LinearRegression()
                        model.fit(X_numeric, y)
                        predictions = model.predict(X_numeric)
                        residuals = y - predictions
                        st.success("✅ Model fitted successfully. Navigate tabs to validate assumptions.")
                    except Exception as e:
                        st.error(f"Error fitting the model: {e}")
                        st.stop()

                    # ---------------------- Tabs ----------------------
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Linearity", "📊 Multicollinearity", "📏 Normality", "📐 Homoscedasticity", "🔁 Autocorrelation"])

                    # ---------------------- Linearity ----------------------
                    with tab1:
                        st.subheader("📈 Residual vs Fitted Plot")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.residplot(x=predictions, y=y, lowess=True, line_kws={"color": "red"}, ax=ax)
                        ax.set_xlabel("Fitted Values")
                        ax.set_ylabel("Residuals")
                        st.pyplot(fig)

                        st.caption("✅ A flat red line near zero indicates that the linearity assumption is likely satisfied.")

                    # ---------------------- Multicollinearity ----------------------
                    with tab2:
                        st.subheader("📊 Variance Inflation Factor (VIF)")
                        if X_numeric.shape[1] > 1:
                            try:
                                vif = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
                                vif_df = pd.DataFrame({"Feature": X_numeric.columns, "VIF": vif})
                                st.dataframe(vif_df)

                                if vif_df["VIF"].max() > 5:
                                    st.error("⚠️ High VIF detected — possible multicollinearity.")
                                else:
                                    st.success("✅ No multicollinearity problem.")
                            except Exception as e:
                                st.error(f"Error computing VIF: {e}")
                        else:
                            st.info("Not enough features to compute VIF.")

                    # ---------------------- Normality ----------------------
                    with tab3:
                        st.subheader("📏 QQ Plot (Normality of Residuals)")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        stats.probplot(residuals, dist="norm", plot=ax)
                        ax.set_title("Normal Q-Q Plot")
                        st.pyplot(fig)

                        st.subheader("🧪 Shapiro-Wilk Test")
                        try:
                            stat, p_value = stats.shapiro(residuals)
                            st.write(f"p-value: {p_value:.4f}")
                            if p_value < 0.05:
                                st.error("Residuals are not normally distributed.")
                            else:
                                st.success("✅ Residuals appear normally distributed.")
                        except Exception as e:
                            st.error(f"Shapiro-Wilk test could not be performed: {e}")

                    # ---------------------- Homoscedasticity ----------------------
                    with tab4:
                        st.subheader("📐 Scale-Location Plot")
                        sqrt_resid = np.sqrt(np.abs(residuals))
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.regplot(x=predictions, y=sqrt_resid, lowess=True, line_kws={"color": "red"}, ax=ax)
                        ax.set_ylabel("√|Standardized Residuals|")
                        ax.set_xlabel("Fitted Values")
                        st.pyplot(fig)

                        st.subheader("🧪 Breusch-Pagan Test")
                        try:
                            _, pval, _, _ = het_breuschpagan(residuals, sm.add_constant(X_numeric))
                            st.write(f"p-value: {pval:.4f}")
                            if pval < 0.05:
                                st.error("Heteroscedasticity detected.")
                            else:
                                st.success("✅ Homoscedasticity holds.")
                        except Exception as e:
                            st.error(f"Breusch-Pagan test failed: {e}")

                    # ---------------------- Autocorrelation ----------------------
                    with tab5:
                        st.subheader("🔁 Residual Plot Over Observations")
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(residuals)
                        ax.set_title("Residuals over Observations")
                        st.pyplot(fig)

                        st.subheader("🧪 Durbin-Watson Test")
                        try:
                            dw = durbin_watson(residuals)
                            st.write(f"Durbin-Watson: {dw:.2f}")
                            if dw < 1.5:
                                st.error("⚠️ Positive autocorrelation detected (DW < 1.5).")
                            elif dw > 2.5:
                                st.error("⚠️ Negative autocorrelation detected (DW > 2.5).")
                            else:
                                st.success("✅ No serious autocorrelation detected (DW ≈ 2).")
                        except Exception as e:
                            st.error(f"Durbin-Watson test failed: {e}")

                        st.subheader("📈 ACF (Autocorrelation Function) Plot")
                        try:
                            fig, ax = plt.subplots(figsize=(6, 3))
                            plot_acf(residuals, ax=ax, lags=20)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"ACF plot failed: {e}")

                        st.caption("""
                        **Interpretation of ACF Plot:**
                        - Each bar shows the autocorrelation at a specific lag.
                        - The blue shaded area is the 95% confidence interval. 
                        - If bars lie outside this region, it indicates significant autocorrelation at that lag.
                        - Ideally, for well-behaved residuals, autocorrelations should lie within the blue area (i.e., be statistically insignificant).
                        """)
