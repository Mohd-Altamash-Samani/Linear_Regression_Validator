# 📊 Linear Regression Assumptions Checker

**Linear_Regression_Validator** is a **Streamlit web app** that helps you validate the key assumptions of a linear regression model — without writing a single line of code.

Upload your dataset, select your variables, and instantly check for:

- ✅ Linearity
- ✅ Multicollinearity
- ✅ Normality of residuals
- ✅ Homoscedasticity
- ✅ Autocorrelation

🌐 **Live App:** [Try it on Streamlit Cloud] - https://linear-regression-validator.streamlit.app/

---

## 🚀 Features

- 📁 Upload CSV or Excel datasets
- 🎯 Select dependent and independent variables
- ❌ Drop irrelevant predictors
- 🔤 Encode categorical variables
- 🧠 Automatically fits a linear regression model
- 📈 Residual vs Fitted plot for linearity
- 📊 VIF calculation for multicollinearity
- 📏 QQ plot + Shapiro-Wilk test for normality
- 📐 Scale-Location plot + Breusch-Pagan test for homoscedasticity
- 🔁 Durbin-Watson test + ACF plot for autocorrelation
